"""Classical estimator tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sensing import ekf, leaky_vel, links
from sim import dynamics as dyn

G = np.array([0.0, 0.0, -dyn.G])
A8, W8 = 1.5, 3.0 / (np.sqrt(2) * 1.5)       # 3 m/s figure-8 (amplitude 1.5 m)


def fig8(t):
    """Kinematic figure-8 drone motion against a HOLD reference (so the error equals the state)."""
    p = np.array([A8 * np.sin(W8 * t), 0.5 * A8 * np.sin(2 * W8 * t), 0.0])
    v = np.array([A8 * W8 * np.cos(W8 * t), A8 * W8 * np.cos(2 * W8 * t), 0.0])
    a = np.array([-A8 * W8 ** 2 * np.sin(W8 * t), -2 * A8 * W8 ** 2 * np.sin(2 * W8 * t), 0.0])
    return p, v, a


def _prm(**kw):
    p = dict(rate=100.0, d0=0.03, sigma_j=0.0, p_drop=0.0, freeze_t=1e9, freeze_dur=0.0, burst_t=1e9,
             burst_dur=0.0, burst_delay=0.0, burst_window=0.0, outage_t=1e9, outage_dur=0.0)
    p.update(kw)
    return {k: jnp.asarray(v, jnp.float32) for k, v in p.items()}


def simulate(mode, n_steps=600, d_hat=0.03, prm_pos=None, prm_vel=None, seed=0, accel_sigma=0.05, q=1.0):
    rng = np.random.default_rng(seed)
    p0, v0, _ = fig8(0.0)
    st = ekf.init(jnp.asarray(p0, jnp.float32), jnp.asarray(v0, jnp.float32))
    lp = links.init(jnp.asarray(p0, jnp.float32), 1.0)
    lv = links.init(jnp.asarray(v0, jnp.float32), 1.0)
    prm_pos, prm_vel = prm_pos or _prm(), prm_vel or _prm()
    sub = jax.jit(ekf.substep)
    prop = jax.jit(ekf.propagate)
    fuse = jax.jit(lambda s, a, t: ekf.fuse_nots(s, {k: ekf.no_sent(x) for k, x in a.items()}, t, d_hat)
                   if mode == 'nots' else ekf.fuse_ts(s, a, t))
    lstep = jax.jit(links.step)
    est, raw, true, stats = [], [], [], []
    I3 = jnp.eye(3)
    for k in range(1, n_steps + 1):
        for i in range(dyn.N_SUB):
            ts = (k - 1) * dyn.DT + (i + 1) * dyn.DT_SUB
            _, _, a = fig8(ts)
            f = a - G + accel_sigma * rng.normal(size=3)
            st = sub(st, I3, jnp.asarray(f, jnp.float32))
        t = k * dyn.DT
        st = prop(st, q)
        p, v, _ = fig8(t)
        tj = jnp.asarray(t, jnp.float32)
        rnd = jnp.asarray([1.0, 0.0, rng.normal()], jnp.float32)
        lp, _, _, _, ap = lstep(lp, prm_pos, tj, jnp.asarray(p, jnp.float32), rnd)
        lv, vout, _, _, av = lstep(lv, prm_vel, tj, jnp.asarray(v, jnp.float32), rnd)
        st = fuse(st, dict(pos=ap, vel=av), tj)
        _, v_hat, _ = ekf.outputs(st)
        est.append(np.asarray(v_hat))
        raw.append(np.asarray(vout))
        true.append(v)
        stats.append(np.asarray(st['n_rejected']).copy())
    return np.array(est), np.array(raw), np.array(true), st, np.array(stats)


@pytest.mark.parametrize('mode', ['ts', 'nots'])
def test_velocity_rmse_halves_raw_delayed(mode):
    est, raw, true, _, _ = simulate(mode, d_hat=0.03)
    s = slice(100, None)
    rmse = lambda x: np.sqrt(np.mean(np.sum((x[s] - true[s]) ** 2, -1)))
    assert rmse(est) <= 0.5 * rmse(raw), (rmse(est), rmse(raw))


def test_ekf_ts_never_fuses_stale_packets():
    """Heavy reordering: every fused packet is at least as new as the last fused one (per channel)."""
    prm = _prm(d0=0.0, sigma_j=0.04)
    p0, v0, _ = fig8(0.0)
    st = ekf.init(jnp.asarray(p0, jnp.float32), jnp.asarray(v0, jnp.float32))
    lv = links.init(jnp.asarray(v0, jnp.float32), 1.0)
    rng = np.random.default_rng(3)
    last, n_out_of_order = -1e9, 0
    for k in range(1, 400):
        t = jnp.asarray(k * dyn.DT, jnp.float32)
        _, v, _ = fig8(k * dyn.DT)
        lv, _, _, _, av = links.step(lv, prm, t, jnp.asarray(v, jnp.float32), jnp.asarray([1.0, 0.0, rng.normal()]))
        before = float(st['last_ts'][1])
        for i in np.where(np.asarray(av['mask']) > 0)[0]:
            n_out_of_order += int(float(av['sent'][i]) < before - 1e-6)
        st = ekf.propagate(st, 1.0)
        st = ekf.fuse_ts(st, dict(pos=dict(av, mask=jnp.zeros_like(av['mask'])), vel=av), t)
        assert float(st['last_ts'][1]) >= last - 1e-6
        last = float(st['last_ts'][1])
    assert n_out_of_order > 0      # the stream really was reordered


def test_ekf_nots_never_reads_timestamps():
    st = ekf.init(jnp.zeros(3), jnp.zeros(3))
    a = dict(val=jnp.zeros((3, 3)), arr=jnp.zeros(3), sent=jnp.zeros(3), mask=jnp.ones(3))
    with pytest.raises(AssertionError):
        ekf.fuse_nots(st, dict(pos=a, vel=a), 0.0, 0.0)
    out1 = ekf.fuse_nots(st, dict(pos=ekf.no_sent(a), vel=ekf.no_sent(a)), 0.0, 0.0)
    b = dict(a, sent=jnp.full(3, 123.0))
    out2 = ekf.fuse_nots(st, dict(pos=ekf.no_sent(b), vel=ekf.no_sent(b)), 0.0, 0.0)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(out1), jax.tree.leaves(out2)))


@pytest.mark.parametrize('mode', ['ts', 'nots'])
def test_chi2_rejects_frozen_velocity_during_acceleration(mode):
    t0, dur = 2.0, 0.5            # around t = 2 s the figure-8 accelerates at ~3-5 m/s^2
    _, _, _, _, rej_none = simulate(mode, n_steps=300)
    _, _, _, _, rej = simulate(mode, n_steps=300, prm_vel=_prm(freeze_t=t0, freeze_dur=dur))
    k0, k1 = int(t0 / dyn.DT), int((t0 + dur) / dyn.DT)
    frozen_packets = k1 - k0
    rejected = (rej[k1, 1] - rej[k0, 1]) - (rej_none[k1, 1] - rej_none[k0, 1])
    assert rejected >= 0.8 * frozen_packets, (rejected, frozen_packets)


def test_repropagation_matches_direct_propagation():
    """Fusing a packet j steps back equals updating the buffered state and re-propagating it with the
    stored inputs; with no packet the buffer is the direct propagation."""
    rng = np.random.default_rng(0)
    st = ekf.init(jnp.asarray([0.1, -0.2, 0.0]), jnp.asarray([0.3, 0.0, -0.1]))
    accs = []
    for k in range(20):
        for i in range(dyn.N_SUB):
            st = ekf.substep(st, jnp.eye(3), jnp.asarray(rng.normal(size=3) - G, jnp.float32))
        accs.append(st['acc'])
        st = ekf.propagate(st, 1.0)
    j = 7
    x_j, P_j = st['x'][j], st['P'][j]
    # no measurement: buffer entry 0 = direct propagation from entry j with the stored inputs
    x0, P0 = ekf.repropagate(x_j, P_j, accs[-j:], 1.0)
    assert np.allclose(x0, st['x'][0], atol=1e-5) and np.allclose(P0, st['P'][0], atol=1e-6)
    # fuse a velocity packet measured j steps ago
    t = 20 * dyn.DT
    z = x_j[:, 1] + jnp.asarray([0.02, -0.01, 0.015], jnp.float32)    # consistent packet (passes the chi^2 gate)
    st2 = ekf.fuse_ts(st, dict(pos=dict(val=jnp.zeros((1, 3)), arr=jnp.zeros(1), sent=jnp.zeros(1), mask=jnp.zeros(1)),
                               vel=dict(val=z[None], arr=jnp.full(1, t), sent=jnp.full(1, t - j * dyn.DT), mask=jnp.ones(1))),
                      jnp.asarray(t, jnp.float32))
    assert float(st2['n_fused'][1]) == float(st['n_fused'][1]) + 1
    S = P_j[1, 1] + ekf.R_VEL
    K = P_j[:, 1] / S
    xu = x_j + (z - x_j[:, 1])[:, None] * K[None, :]
    Pu = P_j - K[:, None] * P_j[1][None, :]
    x0, P0 = ekf.repropagate(xu, Pu, accs[-j:], 1.0)
    assert np.allclose(x0, st2['x'][0], atol=1e-5), np.abs(x0 - st2['x'][0]).max()
    assert np.allclose(P0, st2['P'][0], atol=1e-6)


def test_leaky_velocity_matches_raptor_formula():
    u = leaky_vel.init()
    acc = jnp.asarray([1.0, 0.0, dyn.G])     # level, 1 m/s^2 forward
    for _ in range(100):
        u = leaky_vel.step(u, jnp.eye(3), acc, 0.05)
    a = np.exp(-dyn.DT / 0.05)
    assert np.allclose(float(u[0]), dyn.DT * 1.0 / (1 - a), rtol=1e-3)
    assert abs(float(u[2])) < 1e-5
