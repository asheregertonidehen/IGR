"""Environment and vehicle events.

`sample_events` draws one episode's schedule; `current` evaluates the
time-varying vehicle/environment quantities at a control step. The Dryden-like
turbulence is a first-order low-pass (Ornstein-Uhlenbeck) state advanced per
control step. Severity `sev` scales magnitudes and event probabilities.
"""
import jax
import jax.numpy as jnp

from sim import dynamics as dyn


def _u(key, rng, sev=1.0, clean=0.0):
    """value = clean + sev * (U(lo, hi) - clean)."""
    x = jax.random.uniform(key, minval=rng[0], maxval=rng[1])
    return clean + sev * (x - clean)


def _horizontal_dir(key, vertical=0.1):
    k1, k2 = jax.random.split(key)
    a = jax.random.uniform(k1, maxval=2 * jnp.pi)
    z = jax.random.uniform(k2, minval=-vertical, maxval=vertical)
    v = jnp.stack([jnp.cos(a), jnp.sin(a), z])
    return v / jnp.sqrt(jnp.sum(v * v))


def sample_events(key, spec, sev, drone):
    ks = jax.random.split(key, 20)
    mg = drone['mass'] * dyn.G
    t_lo, t_hi = spec['event_t'][0], spec['event_t'][1]
    ev_time = lambda k: jax.random.uniform(k, minval=t_lo, maxval=t_hi)
    W = _u(ks[0], spec['wind_W'], sev)  # steady |v_w| ~ U(lo, hi) * sev
    wind_steady = W * _horizontal_dir(ks[2])
    turb_tau = jax.random.uniform(ks[3], minval=0.5, maxval=2.0)
    turb_sigma = jax.random.uniform(ks[4]) * spec['turb_frac'] * W
    gust_on = jax.random.uniform(ks[5]) < spec['gust_prob'] * sev
    gust_t = ev_time(ks[6])
    gust_dur = jax.random.uniform(ks[7], minval=spec['gust_dur'][0], maxval=spec['gust_dur'][1])
    gust_peak = _u(ks[8], spec['gust_G'], sev) * mg * _horizontal_dir(ks[9], 0.2)
    pay_on = jax.random.uniform(ks[10]) < spec['payload_prob'] * sev
    sign = jnp.where(jax.random.uniform(ks[11]) < 0.5, -1.0, 1.0)
    pay_dm = sign * _u(ks[12], spec['payload_dm'], sev) * pay_on
    off = jax.random.normal(ks[13], (3,)) * jnp.array([1.0, 1.0, 0.3])
    off = off / jnp.sqrt(jnp.sum(off * off) + 1e-12)
    pay_off = off * jax.random.uniform(ks[14]) * 0.3 * drone['l']
    pay_t = ev_time(ks[15])
    loss_on = jax.random.uniform(ks[16]) < spec['motor_loss_prob'] * sev
    loss_motor = jax.random.randint(ks[17], (), 0, 4)
    loss_k = jnp.where(loss_on, jax.random.uniform(ks[18], minval=spec['motor_loss'][0],
                                                   maxval=spec['motor_loss'][1]), 1.0)
    loss_t = ev_time(ks[19])
    batt = _u(jax.random.fold_in(key, 99), spec['batt_sag'], sev)
    # Optional scripted event (discrimination episodes): exact time / vector.
    return dict(wind_steady=wind_steady, turb_tau=turb_tau, turb_sigma=turb_sigma,
                gust_on=gust_on.astype(jnp.float32), gust_t=gust_t, gust_dur=gust_dur,
                gust_peak=gust_peak, pay_dm=pay_dm, pay_off=pay_off, pay_t=pay_t,
                loss_motor=loss_motor, loss_k=loss_k, loss_t=loss_t, batt=batt,
                ep_len_s=spec['episode_s'])


def apply_script(events, script, drone):
    """Override sampled events with a scripted one (paired episodes).
    script: dict(kind=int, t=float, ...) with jnp scalars; kind 0 = none."""
    mg = drone['mass'] * dyn.G
    t = script['t']
    gust = (script['kind'] == 2) | (script['kind'] == 3)
    lateral = jnp.array([0.0, 1.0, 0.0])
    events = dict(events,
                  gust_on=jnp.where(gust, 1.0, events['gust_on']),
                  gust_t=jnp.where(gust, t, events['gust_t']),
                  gust_dur=jnp.where(gust, 0.6, events['gust_dur']),
                  gust_peak=jnp.where(gust, 0.25 * mg * lateral, events['gust_peak']))
    pay = script['kind'] == 7
    events = dict(events, pay_dm=jnp.where(pay, 0.3, events['pay_dm']),
                  pay_off=jnp.where(pay, jnp.zeros(3), events['pay_off']),
                  pay_t=jnp.where(pay, t, events['pay_t']))
    return events


def init_turbulence():
    return jnp.zeros(3)


def step_turbulence(state, events, noise3):
    a = jnp.exp(-dyn.DT / events['turb_tau'])
    return state * a + events['turb_sigma'] * jnp.sqrt(1 - a * a) * noise3


def current(events, drone, f_ext, turb, t):
    """Time-varying dynamic parameters `dp` and external inputs `ext` at time t."""
    pay = (t >= events['pay_t']).astype(jnp.float32)
    dm = events['pay_dm'] * pay
    k_motor = jnp.where((jnp.arange(4) == events['loss_motor']) & (t >= events['loss_t']),
                        events['loss_k'], 1.0)
    k_batt = 1.0 - events['batt'] * jnp.minimum(t / events['ep_len_s'], 1.0)
    dp = dict(mass=drone['mass'] * (1 + dm), J=drone['J'] * (1 + dm), cf=drone['cf'],
              cm=drone['cm'], T_up=drone['T_up'], T_down=drone['T_down'], l=drone['l'],
              cd=drone['cd'], k_motor=k_motor, k_batt=k_batt,
              payload_mass=dm * drone['mass'], com_offset=events['pay_off'] * pay)
    tg = (t - events['gust_t']) / events['gust_dur']
    in_gust = (tg >= 0) & (tg <= 1) & (events['gust_on'] > 0)
    gust = jnp.where(in_gust, 0.5 * (1 - jnp.cos(2 * jnp.pi * tg)), 0.0) * events['gust_peak']
    ext = dict(f_world=f_ext + gust, v_wind=events['wind_steady'] + turb,
               tau_body=jnp.zeros(3), f_gust=gust)
    return dp, ext
