"""B0 evaluation harness: released RAPTOR through the same sensing-layer outputs.

The spec allows running B0 in numpy on CPU or reimplementing it; it is ported to
JAX (models/raptor_b0.py, parity checked against `foundation_policy` to <1e-6)
so it runs inside the same vmapped evaluator and sees bit-identical sensing
outputs. Inputs are raw SI 22-d RAPTOR observations:
`truth=False` -> [ext_pos, R_hat, ext_vel, gyro, a_prev] (sensing layer),
`truth=True`  -> clean ground truth (convention test).
"""
import jax
import jax.numpy as jnp

from models import inputs as I
from models import raptor_b0
from eval.runner import ModelSet


class B0Arch:
    kind = 'b0'
    cell_type = 'gru'

    def __init__(self, truth=False):
        self.truth = truth

    def init_carry(self):
        return dict(h=jnp.zeros(16))

    def features(self, o, pv, cfg):
        return dict(x=I.raptor_raw_truth(o, pv) if self.truth else I.raptor_raw_layer(o))

    def step(self, p, cfg, carry, ch):
        h, a = raptor_b0.step(p, carry['h'], ch['x'])
        # Return the clipped action as "mean"; the runner applies clip (identity here).
        return dict(h=h), a, dict(hidden=h, h_rep=h)


_ARCHS = {}


def b0_modelset(truth=False):
    p = raptor_b0.load_params()
    if truth not in _ARCHS:
        _ARCHS[truth] = B0Arch(truth)
    arch = _ARCHS[truth]
    params = jax.tree.map(lambda x: x[None], p)
    # RAPTOR resets the GRU to its learned initial state; emulate via carry init.
    arch.init_carry = lambda: dict(h=p['h0'])
    return ModelSet(names=['B0'], arch=arch, params=params, cfg={'input_kind': jnp.zeros(1)}, out='clip')
