"""B0: the released RAPTOR foundation policy (pip `foundation-policy` 1.0.1) in JAX.

Weights are read from the package's checkpoint.h5 (Dense(22->16, ReLU) ->
GRU(16) -> Dense(16->4, identity); 2084 parameters incl. the GRU initial state).
Outputs are clipped to [-1, 1] as L2F does. Parity with the package's numpy
implementation is checked in tests/test_models.py (spec 9.1 allows a
reimplementation if outputs are verified identical on fixed inputs).
Observation: [p - p_ref, R (row-major, body->world), v - v_ref, omega_body, a_prev], raw SI.
"""
import h5py
import importlib_resources
import jax.numpy as jnp
import numpy as np

from models import nn


def checkpoint_path():
    return importlib_resources.files('foundation_policy').joinpath('blob', 'checkpoint.h5')


def load_params():
    with h5py.File(checkpoint_path(), 'r') as f:
        L = f['actor']['layers']
        g = lambda path: np.asarray(L[path][()], dtype=np.float32)
        p = dict(
            d0=dict(w=g('0/weights/parameters').T, b=g('0/biases/parameters')[0]),
            gru=dict(wi=g('1/weights_input/parameters').T, wh=g('1/weights_hidden/parameters').T,
                     bi=g('1/biases_input/parameters'), bh=g('1/biases_hidden/parameters')),
            h0=g('1/initial_hidden_state/parameters'),
            d1=dict(w=g('2/weights/parameters').T, b=g('2/biases/parameters')[0]))
    return {k: (jnp.asarray(v) if not isinstance(v, dict) else {kk: jnp.asarray(vv) for kk, vv in v.items()})
            for k, v in p.items()}


def initial_carry(params):
    return params['h0']


def step(params, h, obs22):
    x = jnp.maximum(nn.dense(params['d0'], obs22), 0.0)
    h = nn.gru(params['gru'], h, x)
    return h, jnp.clip(nn.dense(params['d1'], h), -1.0, 1.0)
