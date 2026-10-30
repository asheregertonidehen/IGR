"""RAPTOR-style student baselines: dense(ReLU) -> GRU(H) -> dense -> tanh.

B-GRU16: H = 16 (RAPTOR student; 2068 params, + 16 for RAPTOR's learned h0).
B-GRU-matched: H chosen so the deployment parameter count on the 54-d native input matches
the Full write-gate cell (computed exactly by `matched_hidden`).
Input kinds (cfg['input_kind'], models/inputs.KINDS): raptor_truth (B1 training), raptor_layer
(B2; B1 deployed), native (B3, B4, A7, A9), raptor_ekf (B5, ekf_nots), ekf_matched (B5m with
ekf_nots, B5m-ts with ekf_ts), raptor_leaky (B6). Inputs are zero-padded to the largest kind so
the variants of a family vmap in one program; padded rows get zero input and zero gradient, and
deploy counts use the real dimension. `heads=True` (A9) adds the training-only representation
heads on the GRU hidden state.
"""
import jax
import jax.numpy as jnp

from models import heads as hd
from models import inputs as I
from models import nn

KIND_INDEX = I.KINDS
INPUT_KIND_NAME = {float(v): k for k, v in I.KINDS.items()}


def gru_param_count(n_in, h):
    return n_in * h + h + 6 * h * h + 6 * h + h * 4 + 4


def matched_hidden(n_in, target):
    return min(range(4, 160), key=lambda h: abs(gru_param_count(n_in, h) - target))


def _pad(x):
    return jnp.concatenate([x, jnp.zeros(I.MAX_IN_DIM - x.shape[0])])


class GRUBaseline:
    kind = 'gru_base'
    cell_type = 'gru'

    def __init__(self, hidden=16, heads=False):
        self.hidden, self.heads = hidden, heads
        self.n_in = I.MAX_IN_DIM

    def init(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        p = dict(d0=nn.dense_init(k1, self.n_in, self.hidden, jnp.sqrt(2.0)),
                 gru=nn.gru_init(k2, self.hidden, self.hidden),
                 out=nn.dense_init(k3, self.hidden, 4, 0.01), log_std=jnp.full(4, -0.5))
        if self.heads:
            p['heads'] = hd.init_heads(k4, self.hidden)
        return p

    def deploy_param_count(self, p=None, cfg=None, input_kind='native'):
        if cfg is not None:
            input_kind = INPUT_KIND_NAME[float(cfg['input_kind'])]
        return gru_param_count(I.KIND_DIMS[input_kind], self.hidden)

    def init_carry(self):
        return dict(h=jnp.zeros(self.hidden))

    def features(self, o, pv, cfg):
        k = cfg['input_kind']
        cands = [_pad(I.raptor_truth(o, pv)), _pad(I.raptor_layer(o)), _pad(I.native_flat(o)),
                 _pad(I.raptor_ekf(o)), _pad(I.ekf_matched(o)), _pad(I.raptor_leaky(o))]
        x = cands[-1]
        for i in range(len(cands) - 2, -1, -1):
            x = jnp.where(k == i, cands[i], x)
        return dict(x=x)

    def step(self, p, cfg, carry, ch):
        x = jax.nn.relu(nn.dense(p['d0'], ch['x']))
        h = nn.gru(p['gru'], carry['h'], x)
        mean = nn.dense(p['out'], h)
        aux = dict(hidden=h, h_rep=h)
        if self.heads:
            aux.update(truth_head=hd.truth_head(p['heads'], h), id_head=hd.id_head(p['heads'], h))
        return dict(h=h), mean, aux
