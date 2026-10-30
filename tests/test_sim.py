"""Acceptance tests for dynamics, IMU, Mahony and links."""
import jax
import jax.numpy as jnp
import numpy as np

from sim import dynamics as dyn
from sim import env as envm
from sim import sampler, specs, tasks
from sensing import links, mahony

jax.config.update('jax_enable_x64', False)


def _drone(i=0):
    d = sampler.sample_drones(8, 0)
    return {k: jnp.asarray(v[i]) for k, v in d.items()}


def _dp(drone, **kw):
    dp = dict(mass=drone['mass'], J=drone['J'], cf=drone['cf'], cm=drone['cm'], T_up=drone['T_up'],
              T_down=drone['T_down'], l=drone['l'], cd=drone['cd'], k_motor=jnp.ones(4), k_batt=1.0,
              payload_mass=0.0, com_offset=jnp.zeros(3))
    dp.update(kw)
    return dp


EXT0 = dict(f_world=jnp.zeros(3), v_wind=jnp.zeros(3), tau_body=jnp.zeros(3), f_gust=jnp.zeros(3))


def _run(dp, sp, steps, ext=EXT0, phys=None):
    phys = phys or dyn.init_phys(jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), sp)

    def body(ph, _):
        ph, f, acc = dyn.substep(ph, dp, ext, sp)
        return ph, (ph['v'], acc, f, ph['w'], ph['rpm'])
    return jax.lax.scan(body, phys, None, length=steps)


def test_hover_keeps_velocity_small():
    for i in range(4):
        drone = _drone(i)
        dp = _dp(drone)
        sp = jnp.full(4, dyn.hover_setpoint(dp))
        _, (v, _, _, _, _) = _run(dp, sp, 2500)  # 5 s
        assert float(jnp.max(jnp.abs(v))) < 1e-3


def test_hover_thrust_differs_across_drones():
    hovers = [float(dyn.hover_setpoint(_dp(_drone(i)))) for i in range(4)]
    assert len(set(np.round(hovers, 4))) == 4


def test_free_fall():
    drone = _drone(1)
    dp = _dp(drone, cf=jnp.zeros(3), cd=0.0)
    _, (_, acc, f, _, _) = _run(dp, jnp.zeros(4), 50)
    assert np.allclose(np.asarray(acc), [0, 0, -9.81], atol=1e-6)
    assert np.allclose(np.asarray(f), 0.0, atol=1e-6)  # accelerometer reads zero in free fall


def test_yaw_sign_cw_motors_positive():
    drone = _drone(2)
    dp = _dp(drone)
    h = dyn.hover_setpoint(dp)
    sp = jnp.array([h, h + 0.05, h, h + 0.05])  # raise CW motors (1, 3)
    phys = dyn.init_phys(jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), sp)
    _, (_, _, _, w, _) = _run(dp, sp, 20, phys=phys)
    assert float(w[-1][2]) > 0  # CW props -> +z (CCW) body reaction torque
    assert abs(float(w[-1][0])) < 1e-6 and abs(float(w[-1][1])) < 1e-6


def test_roll_pitch_sign():
    drone = _drone(3)
    dp = _dp(drone)
    h = dyn.hover_setpoint(dp)
    phys = dyn.init_phys(jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), jnp.full(4, h))
    # Left motors (BL, FL) up -> left side rises -> positive rotation about x (FLU).
    sp = jnp.array([h, h, h + 0.05, h + 0.05])
    _, (_, _, _, w, _) = _run(dp, sp, 20, phys=phys)
    assert float(w[-1][0]) > 0
    # Front motors (FR, FL) up -> nose rises -> negative rotation about y (FLU).
    sp = jnp.array([h + 0.05, h, h, h + 0.05])
    _, (_, _, _, w, _) = _run(dp, sp, 20, phys=phys)
    assert float(w[-1][1]) < 0


def test_motor_step_response_time_constants():
    drone = _drone(0)
    dp = _dp(drone)
    phys = dyn.init_phys(jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), jnp.zeros(4))
    steps = int(round(float(drone['T_up']) / dyn.DT_SUB))
    _, (_, _, _, _, rpm) = _run(dp, jnp.ones(4), steps, phys=phys)
    expected = 1 - np.exp(-steps * dyn.DT_SUB / float(drone['T_up']))
    assert abs(float(rpm[-1][0]) - expected) < 1e-4 and abs(expected - 0.632) < 0.03
    phys = dyn.init_phys(jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]), jnp.zeros(3), jnp.ones(4))
    steps = int(round(float(drone['T_down']) / dyn.DT_SUB))
    _, (_, _, _, _, rpm) = _run(dp, jnp.zeros(4), steps, phys=phys)
    assert abs(float(rpm[-1][0]) - np.exp(-steps * dyn.DT_SUB / float(drone['T_down']))) < 1e-4


def test_imu_at_hover_reads_plus_g():
    drone = _drone(0)
    dp = _dp(drone)
    sp = jnp.full(4, dyn.hover_setpoint(dp))
    _, (_, _, f, _, _) = _run(dp, sp, 10)
    assert np.allclose(np.asarray(f[-1]), [0, 0, 9.81], atol=1e-3)


def test_mahony_converges_from_10_deg():
    q_true = jnp.array([1.0, 0, 0, 0])
    st = mahony.init(q_true, jnp.array([jnp.deg2rad(10.0), 0.0, 0.0]))
    accel = jnp.array([0, 0, 9.81])
    # Error decays as exp(-Kp t); <1 deg after 2 s holds for Kp >= 1.2.
    for _ in range(1000):  # 2 s at 2 ms
        st = mahony.update(st, jnp.zeros(3), accel, mahony.E_X, 2.0, 0.0, dyn.DT_SUB)
    err = 2 * np.degrees(np.arccos(min(1.0, abs(float(st['q'][0])))))
    assert err < 1.0


def test_mahony_yaw_correction():
    st = mahony.init(jnp.array([1.0, 0, 0, 0]), jnp.array([0.0, 0.0, jnp.deg2rad(10.0)]))
    for _ in range(1000):
        st = mahony.update(st, jnp.zeros(3), jnp.array([0, 0, 9.81]), mahony.E_X, 2.0, 0.0, dyn.DT_SUB)
    assert 2 * np.degrees(np.arccos(min(1.0, abs(float(st['q'][0]))))) < 1.0


def _link_prm(**kw):
    p = dict(rate=100.0, d0=0.03, sigma_j=0.0, p_drop=0.0, freeze_t=1e9, freeze_dur=0.0, burst_t=1e9,
             burst_dur=0.0, burst_delay=0.0, burst_window=0.0, outage_t=1e9, outage_dur=0.0)
    p.update(kw)
    return {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}


def _run_link(prm, n, key=0):
    st = links.init(jnp.zeros(1), 1.0)
    rng = np.random.default_rng(key)
    outs, valid = [], []
    for k in range(1, n + 1):
        t = jnp.asarray(k * dyn.DT, jnp.float32)
        rnd = jnp.asarray([rng.uniform(), 0.0, rng.normal()], jnp.float32)
        st, out, v, _, _ = links.step(st, prm, t, jnp.array([float(k)]), rnd)
        outs.append(float(out[0]))
        valid.append(float(v))
    return np.array(outs), np.array(valid)


def test_link_fixed_delay_three_steps():
    outs, valid = _run_link(_link_prm(), 50)
    ks = np.arange(1, 51)
    assert np.all(outs[5:] == ks[5:] - 3)
    assert np.all(valid == 1)


def test_link_reordering_consumed_in_arrival_order():
    outs, _ = _run_link(_link_prm(d0=0.0, sigma_j=0.04), 1000, key=3)
    # out-of-order delivery: the delivered stamp decreases at least once
    assert np.sum(np.diff(outs) < 0) >= 1


def test_link_freeze_holds_value_and_valid():
    outs, valid = _run_link(_link_prm(d0=0.0, freeze_t=0.2, freeze_dur=0.3), 60)
    i0, i1 = 20, 49  # t = 0.21 .. 0.49
    assert np.all(outs[i0:i1] == outs[i0 - 1])
    assert np.all(valid[i0:i1] == 1)


def test_link_drop_invalidates():
    _, valid = _run_link(_link_prm(d0=0.0, p_drop=1.0), 30)
    assert valid[-1] == 0


def test_env_step_runs_and_resets():
    spec, meta = specs.train_spec()
    dset = {k: jnp.asarray(v) for k, v in sampler.sample_drones(16, 5).items()}
    ep = envm.sample_episode(jax.random.PRNGKey(0), dset, spec, 1.0)
    st = envm.reset(ep)
    step = jax.jit(envm.step)
    for i in range(20):
        st, r, term = step(st, jnp.zeros(4), jax.random.PRNGKey(i))
    assert np.isfinite(float(r))
    priv = envm.privileged(st)
    assert priv['descriptor'].shape == (13,)


def test_lookahead_matches_reference():
    spec, _ = specs.train_spec()
    spec = dict(spec, task_probs=jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]))
    task = tasks.sample_task(jax.random.PRNGKey(1), spec)
    look = tasks.lookahead(task, 2.0)
    p2, _ = tasks.ref_state(task, 2.0)
    p26, _ = tasks.ref_state(task, 2.6)
    assert np.allclose(np.asarray(look[6:9]), np.asarray(p26 - p2), atol=1e-5)
    # velocity is the derivative of position
    pa, va = tasks.ref_state(task, 1.0)
    pb, _ = tasks.ref_state(task, 1.0 + 1e-3)
    assert np.allclose(np.asarray((pb - pa) / 1e-3), np.asarray(va), atol=2e-2)
