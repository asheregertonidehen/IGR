"""Quadrotor rigid-body and motor dynamics (RAPTOR / L2F equations).

Conventions (verified against L2F `operations_generic/60_dynamics.h` and the
released parameters):
  * world FLU (x forward, y left, z up), gravity [0, 0, -9.81];
  * angular velocity in the body frame; quaternion [w, x, y, z], body -> world;
  * rotation matrices row-major, R = quat_to_rot(q) maps body to world;
  * motor order [front-right, back-right, back-left, front-left] at
    [+-l/sqrt2, +-l/sqrt2, 0]; spins (CCW, CW, CCW, CW); CW <-> r_tau = +z;
  * motor state rpm in [0, 1]; action a in [-1, 1] -> setpoint (a + 1) / 2.

All functions operate on a single environment and are `vmap`ped by callers.
Integration: semi-implicit Euler at 2 ms (velocity before position, angular
rate before attitude), exact exponential update of the first-order motor lag.
"""
import jax
import jax.numpy as jnp

G = 9.81
DT_SUB = 0.002
N_SUB = 5
DT = DT_SUB * N_SUB
GRAVITY = jnp.array([0.0, 0.0, -G], dtype=jnp.float32)
_S = 1.0 / jnp.sqrt(2.0)
# Unit-arm rotor xy positions; multiply by arm length l.
ROTOR_XY = jnp.array([[_S, -_S], [-_S, -_S], [-_S, _S], [_S, _S]], dtype=jnp.float32)
ROTOR_TORQUE_DIR = jnp.array([-1.0, 1.0, -1.0, 1.0], dtype=jnp.float32)
ACT_BUFFER = 32  # substeps of transport delay that fit the ring buffer (64 ms)


def quat_mul(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_to_rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_exp(rotvec):
    """Unit quaternion for a rotation vector (axis * angle), safe at zero."""
    angle = jnp.sqrt(jnp.sum(rotvec * rotvec) + 1e-24)
    half = 0.5 * angle
    k = jnp.where(angle > 1e-8, jnp.sin(half) / angle, 0.5)
    return jnp.concatenate([jnp.cos(half)[None], k * rotvec])


def quat_integrate(q, omega_body, dt):
    q = quat_mul(q, quat_exp(omega_body * dt))
    return q / jnp.sqrt(jnp.sum(q * q))


def rot6d(R):
    """First two columns of R (6D rotation representation), column-major."""
    return jnp.concatenate([R[:, 0], R[:, 1]])


def tilt_angle(q):
    R = quat_to_rot(q)
    return jnp.arccos(jnp.clip(R[2, 2], -1.0, 1.0))


def motor_thrusts(rpm, dp):
    """Per-motor thrust [N]: k_i * k_batt * sum_j c_fj rpm^j (identical curves)."""
    cf = dp['cf']
    base = cf[0] + cf[1] * rpm + cf[2] * rpm * rpm
    return dp['k_motor'] * dp['k_batt'] * base


def forces_and_torques(phys, dp, ext):
    """Return world-frame acceleration, body-frame angular acceleration and
    body-frame specific force for the current state (no state change)."""
    q, v, w, rpm = phys['q'], phys['v'], phys['w'], phys['rpm']
    R = quat_to_rot(q)
    f = motor_thrusts(rpm, dp)
    thrust_world = R[:, 2] * jnp.sum(f)
    drag = -dp['cd'] * (v - ext['v_wind'])
    force = thrust_world + ext['f_world'] + drag
    acc = force / dp['mass'] + GRAVITY
    # sum_i r_p,i x (0, 0, f_i) with r_p = l/sqrt2 * [(+,-), (-,-), (-,+), (+,+)],
    # written as pairwise differences so symmetric thrust cancels exactly in float32.
    arm = _S * dp['l']
    tau = jnp.stack([arm * ((f[2] - f[1]) + (f[3] - f[0])),
                     -arm * ((f[0] - f[1]) + (f[3] - f[2])),
                     dp['cm'] * ((f[1] - f[0]) + (f[3] - f[2]))])
    # Payload weight acting at the CoM offset (body frame).
    payload_weight_body = R.T @ jnp.array([0.0, 0.0, -dp['payload_mass'] * G])
    tau = tau + jnp.cross(dp['com_offset'], payload_weight_body) + ext['tau_body']
    J = dp['J']
    w_dot = (tau - jnp.cross(w, J * w)) / J
    specific_force_body = R.T @ (force / dp['mass'])
    return acc, w_dot, specific_force_body


def substep(phys, dp, ext, setpoint):
    """One 2 ms step. `setpoint` (4,) in [0, 1] is the motor command already
    passed through the transport-delay buffer. Returns (phys, accel_body)."""
    acc, w_dot, spec_force = forces_and_torques(phys, dp, ext)
    v = phys['v'] + acc * DT_SUB
    p = phys['p'] + v * DT_SUB
    w = phys['w'] + w_dot * DT_SUB
    q = quat_integrate(phys['q'], w, DT_SUB)
    rpm = phys['rpm']
    tau_m = jnp.where(setpoint >= rpm, dp['T_up'], dp['T_down'])
    rpm = setpoint + (rpm - setpoint) * jnp.exp(-DT_SUB / tau_m)
    new = dict(phys, p=p, v=v, w=w, q=q, rpm=jnp.clip(rpm, 0.0, 1.0))
    return new, spec_force, acc


def push_command(act_buf, setpoint, delay_substeps):
    """Shift-register transport delay: returns (buffer, delayed setpoint)."""
    act_buf = jnp.concatenate([setpoint[None], act_buf[:-1]], axis=0)
    idx = jnp.clip(delay_substeps, 0, ACT_BUFFER - 1).astype(jnp.int32)
    return act_buf, act_buf[idx]


def action_to_setpoint(action):
    return (jnp.clip(action, -1.0, 1.0) + 1.0) * 0.5


def hover_setpoint(dp):
    """Normalized motor speed solving sum f_i = m g with all k = 1."""
    cf = dp['cf']
    target = dp['mass'] * G / 4.0 / (dp['k_batt'] * jnp.mean(dp['k_motor']))
    disc = cf[1] ** 2 - 4 * cf[2] * (cf[0] - target)
    return (-cf[1] + jnp.sqrt(jnp.maximum(disc, 0.0))) / (2 * cf[2])


def init_phys(p, v, q, w, rpm):
    return dict(p=p, v=v, q=q, w=w, rpm=rpm,
                act_buf=jnp.tile(rpm[None], (ACT_BUFFER, 1)))
