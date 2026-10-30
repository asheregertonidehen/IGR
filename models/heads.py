"""Training-only representation heads on h_rep. All targets are standardized (7.1b).

truth: current p_err, v_err (world), R (6D), omega, motor states, external force + torque (25)
pred : measured gyro + accel at t+1..t+5 given executed actions a_t..a_{t+4} (30)
id   : current EFFECTIVE r_t2w, log r_t2i, T_up, T_down (4)
The channel predictor P_c (mean + log-variance of each external channel) is part of the cell.
"""
import jax
import jax.numpy as jnp

from models import inputs as I
from models import nn

HEAD_HIDDEN = 64


def init_heads(key, h_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(truth=nn.mlp_init(k1, [h_dim, HEAD_HIDDEN, I.TRUTH_TARGET_DIM]),
                pred=nn.mlp_init(k2, [h_dim + 4 * I.PRED_H, HEAD_HIDDEN, I.PRED_DIM * I.PRED_H]),
                id=nn.mlp_init(k3, [h_dim, HEAD_HIDDEN // 2, I.ID_DIM]))


def truth_head(hp, h):
    return nn.mlp(hp['truth'], h)


def pred_head(hp, h, future_actions):
    return nn.mlp(hp['pred'], jnp.concatenate([h, future_actions], -1))


def id_head(hp, h):
    return nn.mlp(hp['id'], h)
