"""Single-environment episode sampling, reset and 100 Hz step (pure functional JAX).

An episode `ep` bundles everything fixed at reset: drone, constant external
force, event schedule, sensing parameters, reference task, initial state,
mixture category and the teacher truth-channel mask. The env state carries `ep`
so auto-reset can swap in a pre-sampled episode with `jnp.where`.

Suite/training specs are dicts of jnp arrays (traced, so one compiled program
serves every suite). Raw SI quantities only; normalization is in models/inputs.py.
"""
import jax
import jax.numpy as jnp

from sim import dynamics as dyn
from sim import events as evm
from sim import tasks
from sensing import flow, layer

N_HIST = 6  # control-step action history: E's pending-command input and every native student
P_INT_TAU = 2.0   # leaky position-error integral for the expert
P_INT_CLIP = 0.3
EVENT_TAIL_S = 0.5  # a step is an event step inside, or within 0.5 s after, a discrete event
NO_SCRIPT = dict(kind=jnp.asarray(0), t=jnp.asarray(1e9))


def drone_at(dset, idx):
    return {k: v[idx] for k, v in dset.items()}


def sample_episode(key, dset, spec, sev, p_truth=0.0, script=None, drone_idx=None):
    script = NO_SCRIPT if script is None else script
    ks = jax.random.split(key, 8)
    n = dset['mass'].shape[0]
    idx = jax.random.randint(ks[0], (), 0, n) if drone_idx is None else drone_idx
    drone = drone_at(dset, idx)
    cat = jax.random.choice(ks[1], 4, p=spec['mix'])  # clean, sensing, dynamics, both
    sens_on = ((cat == 1) | (cat == 3)).astype(jnp.float32)
    dyn_on = ((cat == 2) | (cat == 3)).astype(jnp.float32)
    f_ext = jax.random.normal(ks[2], (3,)) * drone['sigma_fext'] * drone['mass'] * dyn.G
    events = evm.sample_events(ks[3], spec, sev * dyn_on, drone)
    events = evm.apply_script(events, script, drone)
    task = tasks.sample_task(ks[5], spec)
    peak_t = tasks.peak_accel_time(task, spec['event_t'][0], spec['event_t'][1])
    sens = layer.sample_sensing(ks[4], spec, sev * sens_on, script, peak_t)
    dp0, _ = evm.current(events, drone, f_ext, jnp.zeros(3), 0.0)
    hover = dyn.hover_setpoint(dp0)
    init = tasks.sample_initial_state(ks[6], spec, drone['l'], hover)
    truth_on = (jax.random.uniform(ks[7]) < p_truth).astype(jnp.float32)
    return dict(drone=drone, drone_idx=idx, f_ext=f_ext, events=events, sens=sens, task=task,
                init=init, cat=cat, truth_on=truth_on, script_t=script['t'],
                term_pos=jnp.maximum(20 * drone['l'], 2.0), term_vel=spec['term_vel'])


def _ext_force_torque(phys, dp, ext):
    R = dyn.quat_to_rot(phys['q'])
    drag = -dp['cd'] * (phys['v'] - ext['v_wind'])
    f = ext['f_world'] + drag
    w_body = R.T @ jnp.array([0.0, 0.0, -dp['payload_mass'] * dyn.G])
    tau = jnp.cross(dp['com_offset'], w_body) + ext['tau_body']
    return f, tau


def reset(ep, est=None):
    """`est` (static): None | 'nots' | 'ts' -- which delay-aware KF this program runs."""
    p, v, q, w, rpm = ep['init']
    phys = dyn.init_phys(p, v, q, w, rpm)
    p_ref, v_ref = tasks.ref_state(ep['task'], 0.0)
    look = tasks.lookahead(ep['task'], 0.0)
    ss = layer.init_sensing(ep['sens'], q, p - p_ref, v - v_ref, look, est)
    dp, ext = evm.current(ep['events'], ep['drone'], ep['f_ext'], jnp.zeros(3), 0.0)
    _, _, spec_force = dyn.forces_and_torques(phys, dp, ext)
    R_hat = dyn.quat_to_rot(ss['mahony']['q'])
    zero = jnp.zeros(())
    accel = spec_force + ss['bias'][1]
    obs = dict(gyro=w + ss['bias'][0], accel=accel, att6=dyn.rot6d(R_hat),
               R_hat=R_hat.reshape(-1), ext_pos=p - p_ref, ext_pos_valid=zero + 1, ext_pos_age=zero,
               ext_vel=v - v_ref, ext_vel_valid=zero + 1, ext_vel_age=zero, ref=look,
               ref_valid=zero + 1, ref_age=zero, prev_action=jnp.zeros(4), light=zero + 1,
               prev_actions=jnp.zeros(4 * N_HIST), leaky_v=v - v_ref,
               **layer.est_outputs(ss, est, accel, R_hat))
    return dict(phys=phys, ss=ss, turb=jnp.zeros(3), t=jnp.zeros(()), k=jnp.zeros((), jnp.int32),
                act_hist=jnp.zeros((N_HIST, 4)), obs=obs, ep=ep, alive=jnp.ones((), bool),
                crash_k=jnp.full((), -1, jnp.int32), last_ext=jnp.zeros(6), acc=jnp.zeros(3),
                p_int=jnp.zeros(3), ev_last=jnp.full((), -1e9))


def _in(t, t0, dur):
    return (t >= t0) & (t < t0 + dur)


def event_now(ep, t, obs):
    """Discrete events active at time t. A constant base delay is a
    condition, not an event, so it is excluded. Step changes (payload, motor, actuation
    delay) count at their onset; the 0.5 s tail is added by `ev_last`."""
    sp, ev = ep['sens'], ep['events']
    links_ = [sp['pos'], sp['vel'], sp['ref']]
    sens = jnp.zeros((), bool)
    for lk in links_:
        sens = sens | _in(t, lk['freeze_t'], lk['freeze_dur']) | _in(t, lk['burst_t'], lk['burst_dur'])
    sens = sens | _in(t, sp['vel']['outage_t'], sp['vel']['outage_dur'])
    sens = sens | (obs['ext_pos_valid'] < 0.5) | (obs['ext_vel_valid'] < 0.5) | (obs['ref_valid'] < 0.5)
    sens = sens | ((sp['flow']['is_flow'] > 0) & (obs['light'] < flow.I_MIN))
    onset = lambda t0: _in(t, t0, dyn.DT)
    dyn_ev = (ev['gust_on'] > 0) & _in(t, ev['gust_t'], ev['gust_dur'])
    dyn_ev = dyn_ev | ((jnp.abs(ev['pay_dm']) > 0) & onset(ev['pay_t']))
    dyn_ev = dyn_ev | ((ev['loss_k'] < 1) & onset(ev['loss_t']))
    dyn_ev = dyn_ev | ((sp['act_d1'] != sp['act_d0']) & onset(sp['act_t']))
    return sens | dyn_ev


def with_est(env, est_params):
    """Set the classical-estimator tuning (dict q, dhat, tau of scalars) in an env state or episode."""
    if 'ep' in env:
        return dict(env, ep=with_est(env['ep'], est_params))
    return dict(env, sens=dict(env['sens'], est={k: jnp.asarray(v, jnp.float32) for k, v in est_params.items()}))


def step(state, action, key, est=None):
    """Advance 10 ms. Returns (state, reward, terminated). Crashed (not alive)
    environments are frozen; trainers auto-reset them."""
    ep = state['ep']
    t = state['t']
    sp = ep['sens']
    dp, ext = evm.current(ep['events'], ep['drone'], ep['f_ext'], state['turb'], t)
    setpoint = dyn.action_to_setpoint(action)
    d_act = layer.act_delay_substeps(sp, t)
    noise = jax.random.normal(key, (dyn.N_SUB * layer.N_RND_SUBSTEP + layer.N_RND_CONTROL + 3,))
    rnd_ctl = noise[dyn.N_SUB * layer.N_RND_SUBSTEP:]
    rnd_ctl = rnd_ctl.at[0].set(jax.scipy.stats.norm.cdf(rnd_ctl[0]))
    rnd_ctl = rnd_ctl.at[3].set(jax.scipy.stats.norm.cdf(rnd_ctl[3]))
    rnd_ctl = rnd_ctl.at[6].set(jax.scipy.stats.norm.cdf(rnd_ctl[6]))
    phys, ss = state['phys'], state['ss']
    acc = jnp.zeros(3)
    for i in range(dyn.N_SUB):
        act_buf, delayed = dyn.push_command(phys['act_buf'], setpoint, d_act)
        phys = dict(phys, act_buf=act_buf)
        phys, spec_force, acc_i = dyn.substep(phys, dp, ext, delayed)
        R = dyn.quat_to_rot(phys['q'])
        n9 = noise[i * layer.N_RND_SUBSTEP:(i + 1) * layer.N_RND_SUBSTEP]
        ss = layer.imu_substep(ss, sp, R, phys['w'], spec_force, phys['rpm'], n9, est)
        acc = acc + acc_i / dyn.N_SUB
    t_new = t + dyn.DT
    turb = evm.step_turbulence(state['turb'], ep['events'], rnd_ctl[-3:])
    p_ref, v_ref = tasks.ref_state(ep['task'], t_new)
    look = tasks.lookahead(ep['task'], t_new)
    p_err, v_err = phys['p'] - p_ref, phys['v'] - v_ref
    R = dyn.quat_to_rot(phys['q'])
    ss, obs = layer.control_outputs(ss, sp, t_new, R, p_err, v_err, look, action, rnd_ctl[:-3], est)
    act_hist = jnp.concatenate([action[None], state['act_hist'][:-1]])
    obs['prev_actions'] = act_hist.reshape(-1)     # most recent first: a_t, a_{t-1}, ..., a_{t-5}
    f_ext, tau_ext = _ext_force_torque(phys, dp, ext)
    # Reward (RAPTOR eq. 1), q_z = z component of the unit quaternion [w, x, y, z].
    perr = jnp.sqrt(jnp.sum(p_err ** 2))
    q = phys['q']
    finite = jnp.all(jnp.isfinite(phys['p'])) & jnp.all(jnp.isfinite(q)) & jnp.all(jnp.isfinite(phys['w']))
    crash = (perr > ep['term_pos']) | (jnp.max(jnp.abs(phys['w'])) > 35.0) | ~finite
    term = crash | (jnp.sqrt(jnp.sum(v_err ** 2)) > ep['term_vel'])
    reward = (-perr - 0.2 * jnp.arccos(jnp.clip(1 - jnp.abs(q[3]), -1.0, 1.0))
              - jnp.sqrt(jnp.sum((action - state['obs']['prev_action']) ** 2)) + 1.5 - 100.0 * term)
    decay = jnp.exp(-dyn.DT / P_INT_TAU)
    p_int = jnp.clip(state['p_int'] * decay + p_err * dyn.DT, -P_INT_CLIP, P_INT_CLIP)
    alive = state['alive'] & ~term
    keep = lambda new, old: jax.tree.map(lambda a, b: jnp.where(state['alive'], a, b), new, old)
    phys = keep(phys, state['phys'])
    ev_last = jnp.where(event_now(ep, t_new, obs), t_new, state['ev_last'])
    new = dict(state, phys=phys, ss=keep(ss, state['ss']), turb=turb, t=t_new, k=state['k'] + 1,
               act_hist=act_hist, ev_last=keep(ev_last, state['ev_last']),
               obs=keep(obs, state['obs']), alive=alive,
               crash_k=jnp.where(state['alive'] & term, state['k'] + 1, state['crash_k']),
               last_ext=keep(jnp.concatenate([f_ext, tau_ext]), state['last_ext']),
               acc=keep(acc, state['acc']), p_int=keep(p_int, state['p_int']))
    reward = jnp.where(state['alive'], reward, 0.0)
    return new, reward, term & state['alive']


def privileged(state):
    """Ground truth used by the expert, teacher descriptor, critic and heads (raw SI)."""
    ep, phys, t = state['ep'], state['phys'], state['t']
    dr = ep['drone']
    dp, ext = evm.current(ep['events'], dr, ep['f_ext'], state['turb'], t)
    p_ref, v_ref = tasks.ref_state(ep['task'], t)
    R = dyn.quat_to_rot(phys['q'])
    d_act = jnp.where(t >= ep['sens']['act_t'], ep['sens']['act_d1'], ep['sens']['act_d0'])
    descriptor = jnp.concatenate([
        jnp.stack([jnp.log(dp['mass']), dr['t2w'], jnp.log(dr['t2i']), dr['l'], dr['cm'],
                   dr['T_up'], dr['T_down']]), dp['k_motor'], jnp.stack([dp['k_batt'], d_act])])
    ev = ep['events']
    dyn_vec = jnp.concatenate([ev['wind_steady'], jnp.stack([ev['turb_sigma'], dp['payload_mass'] / dr['mass']]),
                               ext['f_gust'] / (dr['mass'] * dyn.G)])
    # Effective identity: the CURRENT thrust-to-weight and
    # torque-to-inertia, including battery sag, motor weakening and payload (mass and inertia
    # scale by 1 + dm, sim/events.current):
    #   r_t2w_eff = k_batt mean(k_i) T_total / (m(t) g),  r_t2i_eff = k_batt mean(k_i) tau_max / J_xx(t).
    k_eff = dp['k_batt'] * jnp.mean(dp['k_motor'])
    m_fac = dp['mass'] / dr['mass']
    identity = jnp.stack([k_eff * dr['t2w'] / m_fac, jnp.log(k_eff * dr['t2i'] / m_fac), dr['T_up'], dr['T_down']])
    return dict(p_err=phys['p'] - p_ref, R=R.reshape(-1), R6=dyn.rot6d(R), v=phys['v'],
                v_err=phys['v'] - v_ref, w=phys['w'], rpm=phys['rpm'], ext=state['last_ext'],
                v_rel_wind=phys['v'] - ext['v_wind'], look=tasks.lookahead(ep['task'], t),
                descriptor=descriptor, pending=state['act_hist'].reshape(-1),
                corruption=layer.corruption_vector(ep['sens']), dyn_vec=dyn_vec,
                p_int=state['p_int'], mass=dp['mass'], tau_max=jnp.sqrt(2.0) * dr['l'] * dr['t2w'] * dr['mass'] * dyn.G / 4,
                identity=identity, truth_on=ep['truth_on'],
                event_active=((t - state['ev_last']) <= EVENT_TAIL_S + 1e-4).astype(jnp.float32))


def episode_task(key, spec):
    """The reference task `sample_episode(key, ...)` will draw (same key stream)."""
    return tasks.sample_task(jax.random.split(key, 8)[5], spec)
