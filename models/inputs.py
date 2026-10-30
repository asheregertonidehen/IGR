"""Per-architecture input assembly with fixed physical normalization.

Scales: position /1 m, velocity /2 m/s, angular rate /10 rad/s, acceleration
/9.81, lookahead /1 m, forces /(m g), torques /tau_max, actions as-is.
`o` = raw sensing-layer outputs, `pv` = privileged truth (sim/env.privileged).
"""
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

POS, VEL, RATE, ACC, LOOK = 1.0, 2.0, 10.0, 9.81, 1.0
K_HIST = 6                       # last K = 6 commands for every native-sensing variant
HIST_DIM = 4 * K_HIST
LOC_CHANNELS = ('gyro', 'accel', 'att')
EXT_CHANNELS = ('ext_pos', 'ext_vel', 'ref')
EXT_VALUE_DIMS = dict(ext_pos=3, ext_vel=3, ref=9)
# raw channel vectors: value + valid + a zero slot (no timestamps are used)
CH_DIMS = dict(gyro=3, accel=3, att=6, ext_pos=5, ext_vel=5, ref=11, prev_actions=HIST_DIM)
TRUTH_DIM = 19                   # teacher truth channel [p_err, R(9), v_err, w] + mask
DESC_DIM = 13
NATIVE_DIM = 3 + 3 + 6 + 4 + 4 + 10 + HIST_DIM   # 54
RAPTOR_DIM = 22
EKF_MATCHED_DIM = 3 + 3 + 3 + 6 + 3 + HIST_DIM + 10   # 52 (B5m / B5m-ts)
EXPERT_DIM = 84
TRUTH_TARGET_DIM = 25            # [p_err, v_err, R6, w, rpm, f_ext, tau_ext]
TRUTH_SLICES = dict(p_err=slice(0, 3), v_err=slice(3, 6), R6=slice(6, 12), w=slice(12, 15), rpm=slice(15, 19),
                    force=slice(19, 22), torque=slice(22, 25))
PC_DIM = sum(EXT_VALUE_DIMS.values())   # 15: true current value of every external channel
ID_DIM = 4
PRED_DIM = 6                     # gyro + accel per future step
PRED_H = 5
# GRU-baseline input kinds (models/baselines.py)
KINDS = dict(raptor_truth=0, raptor_layer=1, native=2, raptor_ekf=3, ekf_matched=4, raptor_leaky=5)
KIND_DIMS = dict(raptor_truth=RAPTOR_DIM, raptor_layer=RAPTOR_DIM, native=NATIVE_DIM, raptor_ekf=RAPTOR_DIM,
                 ekf_matched=EKF_MATCHED_DIM, raptor_leaky=RAPTOR_DIM)
MAX_IN_DIM = max(KIND_DIMS.values())


def channels(o):
    """Dict of normalized native channels; valid bits appended."""
    return dict(
        gyro=o['gyro'] / RATE,
        accel=o['accel'] / ACC,
        att=o['att6'],
        ext_pos=jnp.concatenate([o['ext_pos'] / POS, o['ext_pos_valid'][None], jnp.zeros(1)]),
        ext_vel=jnp.concatenate([o['ext_vel'] / VEL, o['ext_vel_valid'][None], jnp.zeros(1)]),
        ref=jnp.concatenate([o['ref'] / LOOK, o['ref_valid'][None], jnp.zeros(1)]),
        prev_actions=o['prev_actions'],
    )


def native_flat(o):
    """Flattened native channels + lookahead + K = 6 history for the GRU baselines (54-d, no ages).
    Layout: gyro 0:3, accel 3:6, att 6:12, ext_pos 12:16, ext_vel 16:20, ref 20:30, history 30:54."""
    c = channels(o)
    return jnp.concatenate([c['gyro'], c['accel'], c['att'], c['ext_pos'][:4], c['ext_vel'][:4],
                            c['ref'][:10], c['prev_actions']])


def raptor_truth(o, pv):
    return jnp.concatenate([pv['p_err'] / POS, pv['R'], pv['v_err'] / VEL, pv['w'] / RATE, o['prev_action']])


def raptor_layer(o):
    return jnp.concatenate([o['ext_pos'] / POS, o['R_hat'], o['ext_vel'] / VEL, o['gyro'] / RATE, o['prev_action']])


def raptor_ekf(o):
    """B5: RAPTOR 22-d format built from the ekf_nots estimate."""
    return jnp.concatenate([o['est_p'] / POS, o['R_hat'], o['est_v'] / VEL, o['gyro'] / RATE, o['prev_action']])


def raptor_leaky(o):
    """B6: RAPTOR 22-d with the velocity replaced by RAPTOR's leaky-accelerometer overlay."""
    return jnp.concatenate([o['ext_pos'] / POS, o['R_hat'], o['leaky_v'] / VEL, o['gyro'] / RATE, o['prev_action']])


def ekf_matched(o):
    """B5m / B5m-ts: KF outputs + local sensing + K = 6 history + raw ref_stream with valid (52-d).
    Layout: p 0:3, v 3:6, bias-corrected specific force 6:9, att 9:15, gyro 15:18, history 18:42, ref 42:52."""
    return jnp.concatenate([o['est_p'] / POS, o['est_v'] / VEL, o['est_f'] / ACC, o['att6'], o['gyro'] / RATE,
                            o['prev_actions'], o['ref'] / LOOK, o['ref_valid'][None]])


# velocity slice per input kind (action sensitivity S_v)
VEL_SLICE = {0: slice(12, 15), 1: slice(12, 15), 2: slice(16, 19), 3: slice(12, 15), 4: slice(3, 6), 5: slice(12, 15)}


def raptor_raw_layer(o):
    """B0 (released RAPTOR) expects raw SI inputs."""
    return jnp.concatenate([o['ext_pos'], o['R_hat'], o['ext_vel'], o['gyro'], o['prev_action']])


def raptor_raw_truth(o, pv):
    return jnp.concatenate([pv['p_err'], pv['R'], pv['v_err'], pv['w'], o['prev_action']])


def descriptor(pv):
    d = pv['descriptor']
    mu = jnp.array([-1.0, 3.0, 5.6, 0.1, 0.025, 0.065, 0.165, 1, 1, 1, 1, 1.0, 0.0])
    sd = jnp.array([1.5, 1.5, 1.0, 0.05, 0.015, 0.02, 0.08, 0.2, 0.2, 0.2, 0.2, 0.05, 0.02])
    return (d - mu) / sd


def _ext(pv):
    return jnp.concatenate([pv['ext'][:3] / (pv['mass'] * ACC), pv['ext'][3:] / pv['tau_max']])


def expert(o, pv, corrupt=False):
    """Privileged expert input (84-d; incl. the leaky position-error integral). corrupt=True substitutes the state part with
    sensing-layer outputs (E evaluated through corrupted sensing, gate T)."""
    if corrupt:
        state = [o['ext_pos'] / POS, o['R_hat'], o['ext_vel'] / VEL, o['gyro'] / RATE]
        look = o['ref']
    else:
        state = [pv['p_err'] / POS, pv['R'], pv['v_err'] / VEL, pv['w'] / RATE]
        look = pv['look']
    return jnp.concatenate(state + [o['prev_action'], 2 * pv['rpm'] - 1, _ext(pv), pv['v_rel_wind'] / VEL,
                                    look / LOOK, descriptor(pv), pv['pending'], pv['p_int'] / 0.1])


def truth_channel(pv):
    on = pv['truth_on']
    x = jnp.concatenate([pv['p_err'] / POS, pv['R'], pv['v_err'] / VEL, pv['w'] / RATE])
    return jnp.concatenate([x * on, on[None]])


def critic(o, pv, t_frac):
    return jnp.concatenate([expert(o, pv), pv['corruption'], pv['dyn_vec'], pv['truth_on'][None], t_frac[None]])


CRITIC_DIM = EXPERT_DIM + 21 + 8 + 2


# ---------------------------------------------------------------- head targets
# Every representation target is standardized per dimension with FIXED
# statistics measured once on E_0-driven rollouts of the training distribution
# (train/target_stats.py, RESULTS/target_stats.npz, right after stage E). Fixed statistics (not
# running ones) avoid non-stationary targets; the same file standardizes the targets of every T,
# student and teacher variant, so capability distillation compares like with like.
TARGET_STATS_FILE = 'target_stats.npz'
STD_FLOOR = 0.02   # physical-scale vectors only: ~constant dims (torque without payload offset) are not blown up
STAT_KEYS = ('truth', 'id', 'pred', 'pc')
STAT_DIMS = dict(truth=TRUTH_TARGET_DIM, id=ID_DIM, pred=PRED_DIM, pc=PC_DIM)
_TSTATS = {}


def truth_raw(pv):
    return jnp.concatenate([pv['p_err'] / POS, pv['v_err'] / VEL, pv['R6'], pv['w'] / RATE, 2 * pv['rpm'] - 1, _ext(pv)])


def id_raw(pv):
    return pv['identity']


def pred_raw(o):
    return jnp.concatenate([o['gyro'] / RATE, o['accel'] / ACC])


def pc_raw(pv):
    """True CURRENT value of each external channel in channel units [ext_pos, ext_vel, ref]."""
    return jnp.concatenate([pv['p_err'] / POS, pv['v_err'] / VEL, pv['look'] / LOOK])


def ext_values(ch):
    """Delivered external-channel values (15) in channel units from `channels()` output."""
    return jnp.concatenate([ch['ext_pos'][:3], ch['ext_vel'][:3], ch['ref'][:9]])


def ext_valid(ch):
    return jnp.stack([ch['ext_pos'][3], ch['ext_vel'][3], ch['ref'][9]])


def _stats_path():
    root = Path(__file__).resolve().parents[1]
    return Path(os.environ.get('IGR_RESULTS', root / 'results')) / TARGET_STATS_FILE


def target_stats(required=False):
    """{'truth'|'id'|'pred'|'pc': (mean, std)}. Identity (0, 1) if the file does not exist yet
    (unit tests; NOT cached, so a later load is not shadowed); required=True raises instead."""
    if 'stats' not in _TSTATS:
        p = _stats_path()
        if not p.exists():
            if required:
                raise FileNotFoundError(f'{p} missing: run the target_stats job (after E) first')
            return {k: (np.zeros(d, np.float32), np.ones(d, np.float32)) for k, d in STAT_DIMS.items()}
        # plain numpy constants: the first call may happen inside a jit trace (no tracer may be cached)
        with np.load(p) as f:
            missing = [k for k in STAT_KEYS if f'{k}_mean' not in f]
            if missing:
                raise KeyError(f'{p} lacks {missing}: recompute target_stats')
            _TSTATS['stats'] = {k: (np.asarray(f[f'{k}_mean'], np.float32),
                                    np.maximum(np.asarray(f[f'{k}_std'], np.float32),
                                               STD_FLOOR if k in ('truth', 'pc') else 1e-6).astype(np.float32))
                                for k in STAT_KEYS}
    return _TSTATS['stats']


def standardize(x, k):
    m, s = target_stats()[k]
    return (x - m) / s


def destandardize(y, k):
    m, s = target_stats()[k]
    return y * s + m


def truth_targets(pv):
    return standardize(truth_raw(pv), 'truth')


def truth_destandardize(y):
    """Head output -> physical-scale truth vector (p_err, v_err/2, R6, w/10, 2 rpm - 1, f/(m g), tau/tau_max)."""
    return destandardize(y, 'truth')


def id_targets(pv):
    return standardize(id_raw(pv), 'id')


def pred_signal(o):
    return standardize(pred_raw(o), 'pred')


def pc_targets(pv):
    return standardize(pc_raw(pv), 'pc')
