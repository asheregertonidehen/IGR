"""Expert E: privileged MLP 3x128 ELU on clean full state + drone descriptor.

Gaussian policy in pre-tanh space with learned log-std (init -0.5); the
executed action is tanh(u); deterministic action tanh(mean).
"""
import jax.numpy as jnp

from models import inputs as I
from models import nn


class Expert:
    kind = 'expert'

    def __init__(self, corrupt=False):
        self.corrupt = corrupt  # True: E evaluated through corrupted sensing (gate T)

    def init(self, key):
        return dict(mlp=nn.mlp_init(key, [I.EXPERT_DIM, 128, 128, 128, 4], out_scale=0.01),
                    log_std=jnp.full(4, -0.5))

    def deploy_param_count(self, p, innov_on=False):
        return nn.count_params(p['mlp'])

    def init_carry(self):
        return dict(none=jnp.zeros(1))

    def features(self, o, pv, cfg=None):
        return dict(x=I.expert(o, pv, self.corrupt))

    def step(self, p, cfg, carry, ch):
        return carry, nn.mlp(p['mlp'], ch['x']), dict(hidden=jnp.zeros(1))

    def mean(self, p, x):
        return nn.mlp(p['mlp'], x)
