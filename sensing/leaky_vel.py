"""RAPTOR's own published mitigation for delayed velocity feedback (B6).

RAPTOR supplement Eqs. S1-S4 as implemented in the pinned StampFly policy adapter
(sensor_adaptation/sensoradapt-raptor/docs/raptor_stampfly_deployment.md):
    a_world = R * specific_force_body + gravity_world
    u_k     = exp(-dt / tau) * u_{k-1} + dt * a_world
    v_policy = v_external + u_k
evaluated at the 100 Hz policy rate with the Mahony attitude and the control-rate
accelerometer average. tau is selected on val_id from {0.05, 0.1, 0.2} s.
"""
import jax.numpy as jnp

from sim import dynamics as dyn

GRAVITY = jnp.array([0.0, 0.0, -dyn.G])


def init():
    return jnp.zeros(3)


def step(u, R_hat, accel, tau):
    return jnp.exp(-dyn.DT / tau) * u + dyn.DT * (R_hat @ accel + GRAVITY)
