"""Mahony complementary attitude filter.

Runs every 2 ms substep on simulated gyro/accel. Standard MARG form: the
accelerometer corrects roll/pitch; a magnetometer-like heading reference
(world x-axis seen in the body frame, small noise) corrects yaw only, so yaw is
observable over 20 s episodes.
"""
import jax.numpy as jnp

from sim import dynamics as dyn

E_X = jnp.array([1.0, 0.0, 0.0])
E_Z = jnp.array([0.0, 0.0, 1.0])


def init(q_true, err_rotvec):
    q = dyn.quat_mul(q_true, dyn.quat_exp(err_rotvec))
    return dict(q=q / jnp.sqrt(jnp.sum(q * q)), integral=jnp.zeros(3))


def update(state, gyro, accel, heading_body, kp, ki, dt):
    R_hat = dyn.quat_to_rot(state['q'])
    up_hat = R_hat.T @ E_Z
    a_norm = accel / jnp.sqrt(jnp.sum(accel * accel) + 1e-9)
    e = jnp.cross(a_norm, up_hat)
    h_hat = R_hat.T @ E_X
    h_meas = heading_body - jnp.dot(heading_body, up_hat) * up_hat
    h_meas = h_meas / jnp.sqrt(jnp.sum(h_meas * h_meas) + 1e-9)
    e_h = jnp.cross(h_meas, h_hat)
    e = e + jnp.dot(e_h, up_hat) * up_hat
    integral = state['integral'] + ki * e * dt
    omega = gyro + kp * e + integral
    return dict(q=dyn.quat_integrate(state['q'], omega, dt), integral=integral)
