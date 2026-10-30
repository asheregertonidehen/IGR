"""Sensing / latency layer: maps true simulator state -> what policies see.

Every policy (baselines included) sees its inputs through this layer. Raw SI
outputs are produced here; per-architecture input assembly and fixed physical
normalization live in models/inputs.py.

Channels: gyro(3), accel(3), attitude_est(6D), ext_pos(3)+valid, ext_vel(3)+valid,
ref_stream(9 lookahead)+valid, prev_action(4) and the K = 6 action history (added by
sim/env.py). Ages (A-app-2) are exported for every external channel and zeroed by the
input assembly for other variants.

Classical estimators run here on exactly the same corrupted measurements:
the leaky-velocity overlay (B6) always (cheap), and ONE delay-aware KF mode per compiled
program when the static `est` argument is 'nots' or 'ts' (B5/B5m or B5m-ts). Their
tuning parameters live in the episode's sensing params (`sp['est']`, set per model).
"""
import jax
import jax.numpy as jnp

from sim import dynamics as dyn
from sensing import ekf, flow, imu, leaky_vel, links, mahony

EST_MODES = (None, 'nots', 'ts')
EST_DEFAULT = dict(q=1.0, dhat=0.0, tau=0.05)

# Script kinds for the paired-event episodes.
SCRIPT_NONE, E1_FREEZE, E2_GUST, E3_BOTH, E4_REORDER, E5_REF_LATE, E6_ACT_DELAY, E7_PAYLOAD = range(8)


def _u(key, rng):
    return jax.random.uniform(key, minval=rng[0], maxval=rng[1])


def _sample_link(key, spec, pre, sev, ev_window, peak_t, rate_clean=100.0):
    ks = jax.random.split(key, 12)
    ev = lambda k: _u(k, ev_window)
    rate = rate_clean + sev * (_u(ks[0], spec[f'{pre}_rate']) - rate_clean)
    freeze_on = jax.random.uniform(ks[1]) < spec[f'{pre}_freeze_prob'] * sev
    burst_on = jax.random.uniform(ks[2]) < spec[f'{pre}_burst_prob'] * sev
    # a fraction of the freezes starts at the time of maximum reference acceleration
    # inside the event window (stale data then matters during training); the rest are uniform.
    at_peak = jax.random.uniform(jax.random.fold_in(key, 7)) < spec[f'{pre}_freeze_peak_frac']
    return dict(
        rate=rate, d0=sev * _u(ks[3], spec[f'{pre}_delay']),
        sigma_j=sev * _u(ks[4], spec[f'{pre}_window']) / rate,
        p_drop=sev * _u(ks[5], spec[f'{pre}_drop']),
        freeze_t=jnp.where(freeze_on, jnp.where(at_peak, peak_t, ev(ks[6])), 1e9),
        freeze_dur=_u(ks[7], spec[f'{pre}_freeze_dur']),
        burst_t=jnp.where(burst_on, ev(ks[8]), 1e9), burst_dur=_u(ks[9], spec[f'{pre}_burst_dur']),
        burst_delay=spec[f'{pre}_burst_delay'] + 0.0 * rate,
        burst_window=sev * _u(ks[10], spec[f'{pre}_burst_window']),
        outage_t=jnp.asarray(1e9), outage_dur=jnp.asarray(0.0))


def sample_sensing(key, spec, sev, script, peak_t=None):
    ks = jax.random.split(key, 20)
    ev_window = spec['event_t']
    peak_t = ev_window[0] if peak_t is None else peak_t
    p = dict(imu=imu.sample_imu(ks[0], spec, sev))
    p['kp'] = _u(ks[1], spec['mahony_kp'])
    p['ki'] = _u(ks[2], spec['mahony_ki'])
    axis = jax.random.normal(ks[3], (3,))
    axis = axis / jnp.sqrt(jnp.sum(axis * axis) + 1e-12)
    p['mahony_err'] = axis * jax.random.uniform(ks[4]) * jnp.deg2rad(spec['mahony_init_err_deg']) * sev
    p['pos'] = _sample_link(ks[5], spec, 'ext', sev, ev_window, peak_t)
    vel = _sample_link(ks[6], spec, 'ext', sev, ev_window, peak_t)
    out_on = jax.random.uniform(ks[7]) < spec['vel_outage_prob'] * sev
    vel = dict(vel, outage_t=jnp.where(out_on, _u(ks[8], ev_window), 1e9),
               outage_dur=_u(ks[9], spec['vel_outage_dur']))
    p['ref'] = _sample_link(ks[10], spec, 'ref', sev, ev_window, peak_t, rate_clean=spec['ref_rate'][1])
    p['pos_noise'] = sev * _u(ks[11], spec['mocap_pos_noise'])
    p['vel_noise'] = sev * _u(ks[12], spec['mocap_vel_noise'])
    is_flow = jax.random.uniform(ks[13]) < spec['flow_prob'] * sev
    dip_on = jax.random.uniform(ks[14]) < spec['dark_prob'] * sev
    kf = jax.random.split(ks[15], 4)
    p['flow'] = dict(is_flow=is_flow.astype(jnp.float32), sigma0=_u(kf[0], spec['flow_sigma0']),
                     base=1.0 + sev * (_u(kf[1], spec['flow_light']) - 1.0),
                     dip_on=dip_on.astype(jnp.float32), dip_level=_u(kf[2], spec['dark_level']),
                     dip_t=_u(kf[3], ev_window), dip_dur=_u(ks[16], spec['dark_dur']))
    act0 = sev * _u(ks[17], spec['act_delay'])
    p['act_d0'], p['act_d1'], p['act_t'] = act0, act0, jnp.asarray(1e9)
    p['phases'] = jax.random.uniform(ks[18], (3,))
    # Scripted discrimination events override the sampled schedule.
    kind, t = script['kind'], script['t']
    freeze = (kind == E1_FREEZE) | (kind == E3_BOTH)
    vel = dict(vel, freeze_t=jnp.where(freeze, t, vel['freeze_t']),
               freeze_dur=jnp.where(freeze, 0.5, vel['freeze_dur']))
    reorder = kind == E4_REORDER
    for name, lk in (('pos', p['pos']), ('vel', vel)):
        lk = dict(lk, burst_t=jnp.where(reorder, t, lk['burst_t']),
                  burst_dur=jnp.where(reorder, 1.0, lk['burst_dur']),
                  burst_window=jnp.where(reorder, 6.0, lk['burst_window']),
                  burst_delay=jnp.where(reorder, 0.0, lk['burst_delay']))
        p[name] = lk
    late = kind == E5_REF_LATE
    p['ref'] = dict(p['ref'], burst_t=jnp.where(late, t, p['ref']['burst_t']),
                    burst_dur=jnp.where(late, 1.0, p['ref']['burst_dur']),
                    burst_window=jnp.where(late, 4.0, p['ref']['burst_window']),
                    burst_delay=jnp.where(late, 0.15, p['ref']['burst_delay']))
    act = kind == E6_ACT_DELAY
    p['act_d0'] = jnp.where(act, 0.0, p['act_d0'])
    p['act_d1'] = jnp.where(act, 0.04, p['act_d1'])
    p['act_t'] = jnp.where(act, t, p['act_t'])
    p['est'] = {k: jnp.asarray(v, jnp.float32) for k, v in EST_DEFAULT.items()}
    return p


def act_delay_substeps(sp, t):
    d = jnp.where(t >= sp['act_t'], sp['act_d1'], sp['act_d0'])
    return jnp.round(d / dyn.DT_SUB).astype(jnp.int32)


def init_sensing(sp, q_true, pos_err, vel_err, ref_look, est=None):
    assert est in EST_MODES, est
    ss = dict(
        mahony=mahony.init(q_true, sp['mahony_err']),
        bias=(sp['imu']['gyro_bias'], sp['imu']['accel_bias']),
        pos=links.init(pos_err, sp['phases'][0]),
        vel=links.init(vel_err, sp['phases'][1]),
        ref=links.init(ref_look, sp['phases'][2]),
        gyro_acc=jnp.zeros(3), accel_acc=jnp.zeros(3), leaky=leaky_vel.init(),
    )
    if est is not None:
        ss['ekf'] = ekf.set_aref(ekf.init(pos_err, vel_err), ref_look)
    return ss


def imu_substep(ss, sp, R_true, omega, spec_force, rpm, n9, est=None):
    """One 2 ms IMU sample + Mahony update; accumulates the control-rate average."""
    g, a = imu.substep_reading(sp['imu'], ss['bias'], omega, spec_force, rpm, n9[:6])
    heading = R_true.T @ mahony.E_X + sp['imu']['heading_sigma'] * n9[6:9]
    m = mahony.update(ss['mahony'], g, a, heading, sp['kp'], sp['ki'], dyn.DT_SUB)
    ss = dict(ss, mahony=m, gyro_acc=ss['gyro_acc'] + g, accel_acc=ss['accel_acc'] + a)
    if est is not None:
        ss['ekf'] = ekf.substep(ss['ekf'], dyn.quat_to_rot(m['q']), a)
    return ss


def est_outputs(ss, est, accel, R_hat):
    """Estimator outputs for the policy (zeros when the program runs no KF)."""
    if est is None:
        z = jnp.zeros(3)
        return dict(est_p=z, est_v=z, est_f=z)
    p, v, b = ekf.outputs(ss['ekf'])
    return dict(est_p=p, est_v=v, est_f=accel - R_hat.T @ b)


def control_outputs(ss, sp, t, R_true, pos_err, vel_err, ref_look, prev_action, rnd, est=None):
    """Close one control step: average IMU, run the links, assemble raw channels.
    rnd: 9 uniforms/normals for links (3 per link) + 6 normals for mocap noise
    + 3 normals for flow + 3 normals for bias random walk."""
    gyro = ss['gyro_acc'] / dyn.N_SUB
    accel = ss['accel_acc'] / dyn.N_SUB
    R_hat = dyn.quat_to_rot(ss['mahony']['q'])
    pos_meas = pos_err + sp['pos_noise'] * rnd[9:12]
    pos_st, pos_out, pos_valid, pos_age, pos_arr = links.step(ss['pos'], sp['pos'], t, pos_meas, rnd[0:3])
    fl = sp['flow']
    light = flow.illumination(fl, t)
    is_flow = fl['is_flow'] > 0
    flow_meas = flow.measurement(vel_err, R_true, R_hat) + flow.sigma(fl, light) * rnd[15:18]
    mocap_meas = vel_err + sp['vel_noise'] * rnd[12:15]
    vel_meas = jnp.where(is_flow, flow_meas, mocap_meas)
    blocked = is_flow & (light < flow.I_MIN)
    vel_st, vel_out, vel_valid, vel_age, vel_arr = links.step(ss['vel'], sp['vel'], t, vel_meas, rnd[3:6], blocked)
    ref_st, ref_out, ref_valid, ref_age, _ = links.step(ss['ref'], sp['ref'], t, ref_look, rnd[6:9])
    leaky = leaky_vel.step(ss['leaky'], R_hat, accel, sp['est']['tau'])
    ss = dict(ss, pos=pos_st, vel=vel_st, ref=ref_st, gyro_acc=jnp.zeros(3), accel_acc=jnp.zeros(3),
              bias=imu.step_bias(sp['imu'], ss['bias'], rnd[18:21]), leaky=leaky)
    if est is not None:
        st = ekf.propagate(ss['ekf'], sp['est']['q'])
        arr = dict(pos=pos_arr, vel=vel_arr)
        if est == 'nots':
            st = ekf.fuse_nots(st, {k: ekf.no_sent(v) for k, v in arr.items()}, t, sp['est']['dhat'])
        else:
            st = ekf.fuse_ts(st, arr, t)
        ss['ekf'] = ekf.set_aref(st, ref_out)
    obs = dict(gyro=gyro, accel=accel, att6=dyn.rot6d(R_hat), R_hat=R_hat.reshape(-1),
               ext_pos=pos_out, ext_pos_valid=pos_valid, ext_pos_age=pos_age,
               ext_vel=vel_out, ext_vel_valid=vel_valid, ext_vel_age=vel_age,
               ref=ref_out, ref_valid=ref_valid, ref_age=ref_age, prev_action=prev_action,
               light=light, leaky_v=vel_out + leaky, **est_outputs(ss, est, accel, R_hat))
    return ss, obs


N_RND_CONTROL = 21
N_RND_SUBSTEP = 9


def corruption_vector(sp):
    """Sensing/latency parameters for the asymmetric critic (privileged)."""
    i = sp['imu']
    return jnp.concatenate([
        jnp.stack([sp['pos']['d0'], sp['vel']['d0'], sp['ref']['d0'],
                   sp['pos']['sigma_j'] * sp['pos']['rate'], sp['vel']['sigma_j'] * sp['vel']['rate'],
                   sp['pos']['rate'] / 100.0, sp['vel']['rate'] / 100.0,
                   sp['pos']['p_drop'], sp['vel']['p_drop'], sp['ref']['p_drop'],
                   sp['act_d0'] / 0.02, sp['flow']['is_flow'],
                   i['gyro_sigma'] * 50, i['accel_sigma'] * 3, i['vib_k']]),
        i['gyro_bias'] * 50, i['accel_bias'] * 5])
