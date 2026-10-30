"""Optical-flow velocity source (low-light proxy).

Illumination I(t) is a per-episode base level with one optional dip (linear
0.2 s ramps). Noise sigma = sigma0 / (I + 0.05); no packets are produced while
I < I_MIN. The flow channel reports the world-frame velocity error obtained by
rotating the body-frame measurement through the onboard (Mahony) attitude, so
the ext_vel channel has one frame for all sources.
"""
import jax.numpy as jnp

I_MIN = 0.15
RAMP_S = 0.2


def illumination(fl, t):
    start, end = fl['dip_t'], fl['dip_t'] + fl['dip_dur']
    down = jnp.clip((t - start) / RAMP_S, 0.0, 1.0)
    up = jnp.clip((end - t) / RAMP_S, 0.0, 1.0)
    w = jnp.where(fl['dip_on'] > 0, jnp.minimum(down, up), 0.0)
    return fl['base'] * (1 - w) + fl['dip_level'] * w


def sigma(fl, light):
    return fl['sigma0'] / (light + 0.05)


def measurement(v_err_world, R_true, R_hat):
    """Body-frame velocity measured by flow, re-expressed with the estimated attitude."""
    return R_hat @ (R_true.T @ v_err_world)
