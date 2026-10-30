"""Representation losses shared by the teacher (PPO) and student (distillation)
trainers. Arrays are time-major [T, N, ...]; `done[t]` means the episode ended after step t
(o_{t+1} belongs to a new episode). Every per-step loss takes a weight w [T, N]:
  w_t = (1 + beta e_t) / mean_batch(1 + beta e_t)       (event weighting, normalized to mean 1)
times an optional label mask (unsolved drones) for the losses that imitate a teacher.
"""
import jax
import jax.numpy as jnp

from models import heads as hd
from models import inputs as I

def shift(x, k):
    """y[t] = x[t + k] (zero padded at the end)."""
    if k == 0:
        return x
    return jnp.concatenate([x[k:], jnp.zeros_like(x[:k])], axis=0)


def future_valid(done, k_max=I.PRED_H):
    """valid[t, n, k-1] = t + k < T and no episode boundary in [t, t + k - 1]."""
    T = done.shape[0]
    t_idx = jnp.arange(T)[:, None]
    out, boundary = [], jnp.zeros_like(done, dtype=bool)
    for k in range(1, k_max + 1):
        boundary = boundary | shift(done, k - 1).astype(bool)
        out.append((t_idx + k < T) & ~boundary)
    return jnp.stack(out, axis=-1).astype(jnp.float32)


def future_actions(actions):
    """Teacher-forced executed actions a_t..a_{t+4} (concatenated)."""
    return jnp.concatenate([shift(actions, k) for k in range(I.PRED_H)], axis=-1)


def event_weights(event, beta, mask=None):
    """(1 + beta e_t) normalized to mean 1 over the (unmasked) steps of the batch."""
    w = 1.0 + beta * event
    mask = jnp.ones_like(w) if mask is None else mask
    return mask * w / jnp.maximum(jnp.sum(mask * w) / jnp.maximum(jnp.sum(mask), 1.0), 1e-6)


def wmean(err, w):
    """Weighted mean of a per-step error [T, N] (weights already normalized; masks may zero some)."""
    return jnp.sum(err * w) / jnp.maximum(jnp.sum(jnp.where(w > 0, 1.0, 0.0)), 1.0)


def pred_outputs(heads_p, h_rep, actions):
    return hd.pred_head(heads_p, h_rep, future_actions(actions))


def pred_loss(heads_p, h_rep, actions, signal, done, w):
    """Measured gyro/accel at t+1..t+5 from h_rep_t and a_t..a_{t+4} (standardized)."""
    fut_s = jnp.stack([shift(signal, k) for k in range(1, I.PRED_H + 1)], axis=-2)
    valid = future_valid(done)
    pred = pred_outputs(heads_p, h_rep, actions).reshape(fut_s.shape)
    err = jnp.sum(jnp.mean((pred - fut_s) ** 2, -1) * valid, -1) / jnp.maximum(jnp.sum(valid, -1), 1.0)
    return wmean(err, w * (jnp.sum(valid, -1) > 0)), pred


def mse_loss(out, tgt, w):
    return wmean(jnp.mean((out - tgt) ** 2, -1), w)


def pc_nll(mu, logvar, tgt, w):
    """Gaussian NLL of P_c's mean / log-variance against the TRUE current channel value."""
    nll = 0.5 * (logvar + (tgt - mu) ** 2 * jnp.exp(-logvar))
    return wmean(jnp.mean(nll, -1), w)


def rep_losses(cfg, heads_p, outs, d, w):
    """Student/teacher representation losses (unscaled), dict of scalars.
    outs: h_rep, truth_head, id_head, pc_mu, pc_logvar [T, N, ...]; d: a, sig, done, truth, idt, pc."""
    l_pred, _ = pred_loss(heads_p, outs['h_rep'], d['a'], d['sig'], d['done'], w)
    out = dict(pred=l_pred, truth=mse_loss(outs['truth_head'], d['truth'], w),
               id=mse_loss(outs['id_head'], d['idt'], w))
    if 'pc_mu' in outs:
        out['pc'] = pc_nll(outs['pc_mu'], outs['pc_logvar'], d['pc'], w)
    return out


def capability_losses(cfg, heads_s, outs_s, heads_t, outs_t, actions, done, w):
    """S heads / P_c reproduce T's on the same rollout. Each component is kept only
    if its own loss is active (lam_k > 0), so a leave-one-out ablation also drops its distillation
    term. Returns (heads-and-P_c loss, per-component dict)."""
    sg = jax.lax.stop_gradient
    comp = dict(truth=mse_loss(outs_s['truth_head'], sg(outs_t['truth_head']), w),
                id=mse_loss(outs_s['id_head'], sg(outs_t['id_head']), w))
    valid = future_valid(done)
    pr_s = pred_outputs(heads_s, outs_s['h_rep'], actions)
    pr_t = sg(pred_outputs(heads_t, outs_t['h_rep'], actions))
    sh = pr_s.shape[:-1] + (I.PRED_H, I.PRED_DIM)
    e = jnp.sum(jnp.mean((pr_s.reshape(sh) - pr_t.reshape(sh)) ** 2, -1) * valid, -1) / jnp.maximum(jnp.sum(valid, -1), 1)
    comp['pred'] = wmean(e, w * (jnp.sum(valid, -1) > 0))
    if 'pc_mu' in outs_s:
        comp['pc'] = mse_loss(outs_s['pc_mu'], sg(outs_t['pc_mu']), w)
    on = dict(truth=cfg['lam_truth'], id=cfg['lam_id'], pred=cfg['lam_pred'], pc=cfg['lam_pc'])
    total = sum(jnp.where(on[k] > 0, 1.0, 0.0) * v for k, v in comp.items())
    return total, comp


def r2(out, tgt, mask):
    """Coefficient of determination over masked steps (pooled over dims of standardized targets)."""
    m = mask[..., None]
    n = jnp.maximum(jnp.sum(m), 1.0)
    mean = jnp.sum(tgt * m, (0, 1)) / n
    sse = jnp.sum(((out - tgt) ** 2) * m)
    sst = jnp.sum(((tgt - mean) ** 2) * m)
    return 1.0 - sse / jnp.maximum(sst, 1e-8)
