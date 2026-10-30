"""Reference trajectories, lookahead and initial states.

Langevin references are generated with the random-phase spectral method: a sum
of K sinusoids per axis whose amplitudes follow the power spectrum of RAPTOR's
second-order Langevin process (gamma = 1, omega = 2, sigma = 0.5). This gives
the same position/velocity statistics as the stochastic process while being an
analytic function of time, so exact lookahead and auto-reset are cheap inside
`jit`. All references start at the origin.
"""
import jax
import jax.numpy as jnp

from sim import dynamics as dyn

K = 10
LANGEVIN_GAMMA, LANGEVIN_OMEGA, LANGEVIN_SIGMA = 1.0, 2.0, 0.5
POS_STD = LANGEVIN_SIGMA / jnp.sqrt(2 * LANGEVIN_GAMMA * LANGEVIN_OMEGA ** 2)
LOOKAHEAD_S = jnp.array([0.1, 0.3, 0.6], dtype=jnp.float32)
FIG8_AMPLITUDE = 1.5
FIG8_RAMP_S = 2.0
FIG8_PEAK_ACCEL_PHASE = 0.8169  # phase maximizing |acceleration| of the 1:2 Lissajous
# moderate Langevin = slow Langevin with bandwidth x1.5 (mean |v| 0.82 m/s, p10-p90 0.40-1.27).
HOLD, LANGEVIN, FAST, FIG8, MODERATE = 0, 1, 2, 3, 4
N_KINDS = 5
MODERATE_BW = 1.5
PEAK_GRID = 200


def _langevin_psd(w):
    return LANGEVIN_SIGMA ** 2 / ((LANGEVIN_OMEGA ** 2 - w ** 2) ** 2 + (LANGEVIN_GAMMA * w) ** 2)


def sample_task(key, spec):
    ks = jax.random.split(key, 8)
    kind = jax.random.choice(ks[0], N_KINDS, p=spec['task_probs'])
    w = jax.random.uniform(ks[1], (3, K), minval=0.05, maxval=6.0)
    amp = jnp.sqrt(2 * _langevin_psd(w) * (6.0 - 0.05) / K)
    amp = amp * POS_STD / jnp.sqrt(jnp.sum(amp ** 2, axis=1, keepdims=True) / 2)
    phase = jax.random.uniform(ks[2], (3, K), maxval=2 * jnp.pi)
    bw = jnp.where(kind == FAST, jax.random.uniform(ks[3], minval=spec['fast_bw'][0],
                                                    maxval=spec['fast_bw'][1]),
                   jnp.where(kind == MODERATE, MODERATE_BW, 1.0))
    speed = jax.random.uniform(ks[4], minval=spec['fig8_speed'][0], maxval=spec['fig8_speed'][1])
    omega = speed / (jnp.sqrt(2.0) * FIG8_AMPLITUDE)
    lock_t = spec['fig8_phase_lock_t']
    random_phase = jax.random.uniform(ks[5], maxval=2 * jnp.pi)
    locked = FIG8_PEAK_ACCEL_PHASE - omega * (lock_t - FIG8_RAMP_S / 2)
    phi0 = jnp.where(lock_t > 0, locked, random_phase)
    yaw = jax.random.uniform(ks[6], maxval=2 * jnp.pi)
    return dict(kind=kind, w=w * bw, amp=amp, phase=phase, f8_omega=omega,
                f8_phi0=phi0, f8_yaw=yaw)


def _fig8_phase(task, t):
    om, tr = task['f8_omega'], FIG8_RAMP_S
    phi = jnp.where(t < tr, om * t * t / (2 * tr), om * (t - tr / 2))
    dphi = jnp.where(t < tr, om * t / tr, om)
    return phi, dphi


def ref_state(task, t):
    """Reference position and velocity at time t (seconds)."""
    arg = task['w'] * t + task['phase']
    p_lang = jnp.sum(task['amp'] * (jnp.sin(arg) - jnp.sin(task['phase'])), axis=1)
    v_lang = jnp.sum(task['amp'] * task['w'] * jnp.cos(arg), axis=1)
    phi, dphi = _fig8_phase(task, t)
    a, p0 = FIG8_AMPLITUDE, task['f8_phi0']
    x = a * (jnp.sin(phi + p0) - jnp.sin(p0))
    y = 0.5 * a * (jnp.sin(2 * (phi + p0)) - jnp.sin(2 * p0))
    vx = a * jnp.cos(phi + p0) * dphi
    vy = a * jnp.cos(2 * (phi + p0)) * dphi
    c, s = jnp.cos(task['f8_yaw']), jnp.sin(task['f8_yaw'])
    p_f8 = jnp.stack([c * x - s * y, s * x + c * y, 0.0])
    v_f8 = jnp.stack([c * vx - s * vy, s * vx + c * vy, 0.0])
    kind = task['kind']
    is_lang = (kind == LANGEVIN) | (kind == FAST) | (kind == MODERATE)
    p = jnp.where(is_lang, p_lang, jnp.where(kind == FIG8, p_f8, 0.0))
    v = jnp.where(is_lang, v_lang, jnp.where(kind == FIG8, v_f8, 0.0))
    return p, v


def ref_accel(task, t):
    """Analytic reference acceleration at time t (3,)."""
    arg = task['w'] * t + task['phase']
    a_lang = -jnp.sum(task['amp'] * task['w'] ** 2 * jnp.sin(arg), axis=1)
    phi, dphi = _fig8_phase(task, t)
    ddphi = jnp.where(t < FIG8_RAMP_S, task['f8_omega'] / FIG8_RAMP_S, 0.0)
    a, p0 = FIG8_AMPLITUDE, task['f8_phi0']
    s1, c1 = jnp.sin(phi + p0), jnp.cos(phi + p0)
    s2, c2 = jnp.sin(2 * (phi + p0)), jnp.cos(2 * (phi + p0))
    ax = a * (-s1 * dphi ** 2 + c1 * ddphi)
    ay = 0.5 * a * (-4 * s2 * dphi ** 2 + 2 * c2 * ddphi)
    c, s = jnp.cos(task['f8_yaw']), jnp.sin(task['f8_yaw'])
    a_f8 = jnp.stack([c * ax - s * ay, s * ax + c * ay, 0.0])
    kind = task['kind']
    is_lang = (kind == LANGEVIN) | (kind == FAST) | (kind == MODERATE)
    return jnp.where(is_lang, a_lang, jnp.where(kind == FIG8, a_f8, 0.0))


def peak_accel_time(task, t0, t1):
    """Time of maximum |reference acceleration| in [t0, t1] (grid of PEAK_GRID points).
    Used for maneuver-placed training freezes and the primary paired-event protocol.
    A hold reference has no maneuver; it returns t0."""
    grid = t0 + (t1 - t0) * jnp.arange(PEAK_GRID) / (PEAK_GRID - 1)
    acc = jax.vmap(lambda s: ref_accel(task, s))(grid)
    return grid[jnp.argmax(jnp.sum(acc * acc, axis=-1))]


def lookahead(task, t):
    """Reference offsets at +0.1, +0.3, +0.6 s relative to the current reference (9,)."""
    p_now, _ = ref_state(task, t)
    fut = jax.vmap(lambda dt: ref_state(task, t + dt)[0])(LOOKAHEAD_S)
    return (fut - p_now[None]).reshape(-1)


def _random_unit(key):
    v = jax.random.normal(key, (3,))
    return v / jnp.sqrt(jnp.sum(v * v) + 1e-12)


def sample_initial_state(key, spec, drone_l, hover_rpm):
    """RAPTOR initial state: position up to 10 l, attitude up to 90 deg, velocity
    up to 1 m/s, angular velocity up to 1 rad/s, 10 % of episodes at target.
    Eval-only `midair` suites override the speed range."""
    ks = jax.random.split(key, 9)
    at_target = jax.random.uniform(ks[0]) < spec['init_at_target']
    pos = _random_unit(ks[1]) * jax.random.uniform(ks[2]) * spec['init_pos_l'] * drone_l
    angle = jax.random.uniform(ks[3]) * spec['init_att_max']
    q = dyn.quat_exp(_random_unit(ks[4]) * angle)
    speed = jax.random.uniform(ks[5], minval=spec['init_vel'][0], maxval=spec['init_vel'][1])
    vel = _random_unit(ks[6]) * speed
    omega = _random_unit(ks[7]) * jax.random.uniform(ks[8]) * spec['init_angvel_max']
    ident = jnp.array([1.0, 0.0, 0.0, 0.0])
    z3 = jnp.zeros(3)
    pos, vel, omega = (jnp.where(at_target, z3, x) for x in (pos, vel, omega))
    q = jnp.where(at_target, ident, q)
    rpm = jnp.full(4, hover_rpm)
    return pos, vel, q, omega, rpm
