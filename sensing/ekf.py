"""Delay-aware Kalman filter for the classical baselines B5 / B5m / B5m-ts.

Two modes share one implementation:
  * `ekf_nots` (primary, same information as Full): no timestamps. A packet delivered at
    control step t is assumed to have been measured a fixed nominal delay d_hat earlier
    (t - d_hat), as autopilots are configured. Jitter, reordering and other delays are
    not compensated; only chi^2 gating can catch them. `fuse` is called with an arrival
    stream that has NO 'sent' key, and asserts it (information-parity guarantee).
  * `ekf_ts` (privileged reference): fuses each packet at its true send time and
    discards packets older than the last fused send time of that channel.

State per world axis: (p_err, v_err, b) with b a world-frame accelerometer bias
(random walk). Attitude is the Mahony estimate every variant uses. Prediction per 2 ms
substep with the bias-corrected, gravity-compensated accelerometer rotated by the Mahony
attitude, minus the reference acceleration fitted to the delivered lookahead (the state is
the error relative to the moving reference, like the channels).

Rewind and re-propagate. The transition is time-invariant and identical for the three
axes, so the covariance is one 3x3 matrix shared by the axes and the effect of fusing a
packet at a buffered past step j on every later buffered step i is exactly
    dx_i = Phi^(j-i) dx_j,   dP_i = Phi^(j-i) dP_j Phi^(j-i)^T,
with Phi the one-control-step transition (process noise is additive). This equals
rewinding to step j, applying the update and re-propagating with the stored inputs
(tested to 1e-5), with fixed-size masked updates and no Python control flow. The ring
buffer holds NB = 32 control steps (310 ms >= the 300 ms required).

Outlier gating: chi^2 at 99 % on the 3-axis innovation (11.34). A channel whose packets
have all been rejected for LOCKOUT_S accepts the next packet unconditionally (standard
estimator practice against permanent lock-out after divergence).
"""
import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm

from sim import dynamics as dyn

NB = 32
CHI2_3_99 = 11.345
LOCKOUT_S = 1.0
GRAVITY = np.array([0.0, 0.0, -dyn.G], np.float32)
# Fixed noise parameters at the midpoint of the training sensing ranges (configs/suites.yaml):
R_POS = 0.003 ** 2            # mocap position noise U(0.001, 0.005) m
R_VEL = 0.035 ** 2            # mocap U(0.01, 0.05) / flow sigma0 U(0.02, 0.05) m/s
Q_ACC = 1e-3                 # accel white-noise PSD [m^2/s^3] (control-rate IMU noise + vibration)
Q_BIAS = 1e-3                # world-frame bias random walk [m^2/s^5] (accel bias is constant per episode; rotation and attitude error move it slowly)
P0 = np.diag([0.01 ** 2, 0.05 ** 2, 0.3 ** 2]).astype(np.float32)
# least-squares fit of the delivered lookahead offsets d(tau) = v tau + a tau^2 / 2
_TAU = np.array([0.1, 0.3, 0.6])
_X = np.stack([_TAU, 0.5 * _TAU ** 2], 1)
AREF_FIT = np.linalg.pinv(_X)[1].astype(np.float32)        # (3,) weights over the 3 horizons


def _matrices():
    dt = dyn.DT_SUB
    F = np.array([[1, dt, -dt * dt / 2], [0, 1, -dt], [0, 0, 1]])
    B = np.array([dt * dt / 2, dt, 0.0])
    A = np.array([[0, 1, 0], [0, 0, -1], [0, 0, 0.0]])

    def q_sub(G):
        # Van Loan discretization of x' = A x + G w (unit PSD)
        M = np.zeros((6, 6))
        M[:3, :3], M[:3, 3:], M[3:, 3:] = -A, G @ G.T, A.T
        E = expm(M * dt)
        Fd = E[3:, 3:].T
        return Fd @ E[:3, 3:]
    qa, qb = q_sub(np.array([[0.0], [1.0], [0.0]])), q_sub(np.array([[0.0], [0.0], [1.0]]))
    Fc, Qa, Qb = np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))
    for _ in range(dyn.N_SUB):
        Fc = F @ Fc
        Qa, Qb = F @ Qa @ F.T + qa, F @ Qb @ F.T + qb
    phi = np.stack([np.linalg.matrix_power(Fc, n) for n in range(NB)])
    f32 = lambda x: np.asarray(x, np.float32)
    return f32(F), f32(B), f32(Fc), f32(Qa), f32(Qb), f32(phi)


F_SUB, B_SUB, F_CTL, QA_CTL, QB_CTL, PHI = _matrices()
H_IDX = (0, 1)   # channel 0 = ext_pos measures p, channel 1 = ext_vel measures v
R_CH = (R_POS, R_VEL)


def init(pos_err, vel_err):
    x0 = jnp.stack([pos_err, vel_err, jnp.zeros(3)], axis=-1)          # [axis, state]
    return dict(x=jnp.broadcast_to(x0, (NB, 3, 3)), P=jnp.broadcast_to(jnp.asarray(P0), (NB, 3, 3)),
                acc=jnp.zeros((3, 3)), aref=jnp.zeros(3), last_ts=jnp.full(2, -1e9),
                last_acc=jnp.zeros(2), n_fused=jnp.zeros(2), n_rejected=jnp.zeros(2))


def substep(st, R_hat, accel_meas):
    """Accumulate one 2 ms substep's input effect (gravity-compensated world accel minus
    the fitted reference acceleration) into the current control step."""
    u = R_hat @ accel_meas + GRAVITY - st['aref']
    return dict(st, acc=st['acc'] @ F_SUB.T + u[:, None] * B_SUB[None, :])


def propagate(st, q_scale):
    """Close a control step: new 'now' entry = F_ctl x_prev + accumulated inputs; shift buffer."""
    x_new = st['x'][0] @ F_CTL.T + st['acc']
    P_new = F_CTL @ st['P'][0] @ F_CTL.T + q_scale * (Q_ACC * QA_CTL + Q_BIAS * QB_CTL)
    return dict(st, x=jnp.concatenate([x_new[None], st['x'][:-1]]),
                P=jnp.concatenate([P_new[None], st['P'][:-1]]), acc=jnp.zeros((3, 3)))


def _fuse_one(st, c, z, t_meas, ok, t):
    """Fuse one packet of channel c measured at t_meas (buffered step j)."""
    h = H_IDX[c]
    j = jnp.clip(jnp.round((t - t_meas) / dyn.DT).astype(jnp.int32), 0, NB)
    ok = ok & (j < NB)
    jj = jnp.minimum(j, NB - 1)
    Pj, xj = st['P'][jj], st['x'][jj]
    S = Pj[h, h] + R_CH[c]
    K = Pj[:, h] / S                                    # (3,) shared by the axes
    nu = z - xj[:, h]                                   # (3 axes,)
    nis = jnp.sum(nu * nu) / S
    accept = ok & ((nis <= CHI2_3_99) | (t - st['last_acc'][c] > LOCKOUT_S))
    dx = nu[:, None] * K[None, :]                       # [axis, state]
    dP = -K[:, None] * Pj[h][None, :]
    n = jj - jnp.arange(NB)
    sel = ((n >= 0) & accept).astype(jnp.float32)
    Phi = jnp.asarray(PHI)[jnp.clip(n, 0, NB - 1)]      # [NB, 3, 3]
    x = st['x'] + sel[:, None, None] * jnp.einsum('ist,at->ias', Phi, dx)
    P = st['P'] + sel[:, None, None] * jnp.einsum('ist,tu,ivu->isv', Phi, dP, Phi)
    P = 0.5 * (P + jnp.swapaxes(P, 1, 2))
    last_ts = st['last_ts'].at[c].set(jnp.where(accept, jnp.maximum(st['last_ts'][c], t_meas), st['last_ts'][c]))
    last_acc = st['last_acc'].at[c].set(jnp.where(accept, t, st['last_acc'][c]))
    n_f = st['n_fused'].at[c].add(accept.astype(jnp.float32))
    n_r = st['n_rejected'].at[c].add((ok & ~accept).astype(jnp.float32))
    return dict(st, x=x, P=P, last_ts=last_ts, last_acc=last_acc, n_fused=n_f, n_rejected=n_r), accept & ok


def fuse_nots(st, arrivals, t, d_hat):
    """ekf_nots: `arrivals` = {'pos'|'vel': dict(val, arr, mask)}; send times are NOT available."""
    for c, name in enumerate(('pos', 'vel')):
        a = arrivals[name]
        assert 'sent' not in a, 'ekf_nots must never read packet send timestamps'
        for k in range(a['val'].shape[0]):
            st, _ = _fuse_one(st, c, a['val'][k], t - d_hat, a['mask'][k] > 0, t)
    return st


def fuse_ts(st, arrivals, t):
    """ekf_ts: fuse at the true send time; drop packets older than the last fused one."""
    for c, name in enumerate(('pos', 'vel')):
        a = arrivals[name]
        for k in range(a['val'].shape[0]):
            fresh = a['sent'][k] >= st['last_ts'][c] - 1e-6
            st, _ = _fuse_one(st, c, a['val'][k], a['sent'][k], (a['mask'][k] > 0) & fresh, t)
    return st


def set_aref(st, ref_look):
    """Reference acceleration fitted to the delivered lookahead offsets (9,) = [horizon, axis]."""
    return dict(st, aref=jnp.asarray(AREF_FIT) @ ref_look.reshape(3, 3))


def outputs(st):
    x = st['x'][0]
    return x[:, 0], x[:, 1], x[:, 2]      # p_err_hat, v_err_hat, world-frame accel bias


def no_sent(arr):
    """Strip send timestamps (the separate data path for ekf_nots)."""
    return {k: v for k, v in arr.items() if k != 'sent'}


# brute-force reference used by the tests: direct re-propagation of the stored inputs
def repropagate(x_j, P_j, inputs, q_scale):
    """Propagate (x_j, P_j) through a list of per-control-step accumulated inputs (oldest first)."""
    x, P = x_j, P_j
    for acc in inputs:
        x = x @ F_CTL.T + acc
        P = F_CTL @ P @ F_CTL.T + q_scale * (Q_ACC * QA_CTL + Q_BIAS * QB_CTL)
    return x, P

