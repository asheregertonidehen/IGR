"""Training diagnostics, logged every ~50 updates for every model with heads:
  * cosine between the control-loss gradient and each representation-loss gradient, with
    respect to the representation-module parameters (the humanoid negative result found
    |cos| < 0.01);
  * per-head R^2 on the held-in minibatch, split by clean and event steps;
  * each representation loss (times its weight) relative to the action / control loss.
"""
import jax
import jax.numpy as jnp

from models import gated_cell
from train import losses

DIAG_EVERY_UPDATES = 50


def rep_subtree(params):
    """Representation-module parameters: the gated cell's module (GRU_rep, P_c, E_nu, gates, W_corr,
    r) or, for a plain GRU with heads (A9), the shared trunk (input layer + GRU)."""
    keys = [k for k in gated_cell.REP_MODULE if k in params and k != 'heads']
    if not keys:
        keys = [k for k in ('d0', 'gru') if k in params]
    return {k: params[k] for k in keys}


def _flat(tree):
    return jnp.concatenate([x.reshape(-1) for x in jax.tree.leaves(tree)])


def cosines(control_fn, rep_fns, params):
    """control_fn(params) -> scalar; rep_fns {name: fn(params) -> scalar}. Returns {name: cos}."""
    g_c = _flat(rep_subtree(jax.grad(control_fn)(params)))
    out = {}
    for name, fn in rep_fns.items():
        g = _flat(rep_subtree(jax.grad(fn)(params)))
        out[f'cos_{name}'] = jnp.sum(g_c * g) / jnp.maximum(jnp.linalg.norm(g_c) * jnp.linalg.norm(g), 1e-12)
    return out


def head_r2(outs, d):
    """R^2 of the truth / identity / P_c heads, split by clean (e_t = 0) and event (e_t = 1) steps."""
    ev = d['ev']
    out = {}
    pairs = [('truth', 'truth_head', 'truth'), ('id', 'id_head', 'idt'), ('pc', 'pc_mu', 'pc')]
    for name, ok, tk in pairs:
        if ok in outs and tk in d:
            out[f'r2_{name}_clean'] = losses.r2(outs[ok], d[tk], 1.0 - ev)
            out[f'r2_{name}_event'] = losses.r2(outs[ok], d[tk], ev)
    if 'truth_head' in outs and 'truth' in d:
        sl = slice(3, 6)   # velocity error dims of the 25-d truth vector
        out['r2_vel_clean'] = losses.r2(outs['truth_head'][..., sl], d['truth'][..., sl], 1.0 - ev)
        out['r2_vel_event'] = losses.r2(outs['truth_head'][..., sl], d['truth'][..., sl], ev)
    return out
