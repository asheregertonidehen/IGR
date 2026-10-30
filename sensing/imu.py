"""IMU model.

Noise sigmas are defined for the control-rate (10 ms averaged) reading; the
2 ms substep samples that feed the Mahony filter carry sigma * sqrt(5) so that
their average has the specified sigma. Bias random walk advances per control
step. Severity `sev` scales every corruption magnitude (0 = ideal IMU).
"""
import jax
import jax.numpy as jnp

from sim import dynamics as dyn

GYRO_SAT = 35.0
ACCEL_SAT = 16 * dyn.G


def sample_imu(key, spec, sev):
    ks = jax.random.split(key, 8)
    u = lambda k, r: jax.random.uniform(k, minval=r[0], maxval=r[1])
    sym = lambda k: jax.random.uniform(k, (3,), minval=-1.0, maxval=1.0)
    bias_mult = spec['imu_bias_mult']
    return dict(
        gyro_sigma=sev * u(ks[0], spec['gyro_noise']),
        gyro_bias=sev * bias_mult * spec['gyro_bias'] * sym(ks[1]),
        gyro_rw=sev * spec['gyro_rw'],
        gyro_scale=1 + sev * spec['imu_scale_err'] * sym(ks[2]),
        accel_sigma=sev * spec['accel_noise_mult'] * u(ks[3], spec['accel_noise']),
        accel_bias=sev * bias_mult * spec['accel_bias'] * sym(ks[4]),
        accel_scale=1 + sev * spec['imu_scale_err'] * sym(ks[5]),
        vib_k=sev * spec['vib_mult'] * u(ks[6], spec['vib_k']),
        heading_sigma=sev * 0.02,
    )


def substep_reading(imu, bias, omega, spec_force, rpm, n6):
    """Noisy gyro/accel samples at the 2 ms rate. `bias` = (gyro_bias, accel_bias)
    current values (with random walk)."""
    root = jnp.sqrt(float(dyn.N_SUB))
    vib = imu['vib_k'] * jnp.mean(rpm) ** 2
    a_sigma = jnp.sqrt(imu['accel_sigma'] ** 2 + vib ** 2)
    gyro = imu['gyro_scale'] * omega + bias[0] + imu['gyro_sigma'] * root * n6[:3]
    accel = imu['accel_scale'] * spec_force + bias[1] + a_sigma * root * n6[3:]
    return jnp.clip(gyro, -GYRO_SAT, GYRO_SAT), jnp.clip(accel, -ACCEL_SAT, ACCEL_SAT)


def step_bias(imu, bias, n3):
    return (bias[0] + imu['gyro_rw'] * n3, bias[1])
