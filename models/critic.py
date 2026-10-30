"""Asymmetric critic: non-recurrent MLP 3x128 on full privileged state,
drone descriptor and corruption/event parameters."""
from models import inputs as I
from models import nn


def init(key, n_in=I.CRITIC_DIM):
    return nn.mlp_init(key, [n_in, 128, 128, 128, 1], out_scale=1.0)


def value(p, x):
    return nn.mlp(p, x)[..., 0]
