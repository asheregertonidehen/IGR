"""Packet delivery links with delay, jitter, drops, freezes and NO FIFO.

on send at t_s (every 1/f s, accumulated phase at control-step resolution):
    if rand < p_drop (or outage / blocked): discard
    arrival = t_s + d0 + |N(0, sigma_j)|      (independent per packet -> reordering)
each control step at t:
    delivered = packets with arrival <= t, processed in ARRIVAL order;
    channel value = value of the packet that arrived LAST (not the newest stamp);
    valid = a packet was delivered within T_VALID; during a freeze the value is
    held and valid stays 1 (silent stuck sensor).
Delay bursts (delay_burst suite, E4/E5 scripts) add base delay and/or jitter
for a time window. Reorder window ~= sigma_j * f packets.

`step` also returns the per-step ARRIVAL STREAM used by the classical
EKF baselines: up to MAX_ARRIVALS packets delivered this step, in arrival order,
with their value, arrival time and send time. During a freeze each delivered packet
carries the held (stale) value, i.e. the stuck sensor keeps sending stale data.
The send time is kept in its own key so `ekf_nots` can be given a stream without it.
"""
import jax
import jax.numpy as jnp

from sim import dynamics as dyn

CAP = 32
T_VALID = 0.1
EPS_T = 1e-4
MAX_ARRIVALS = 3   # packets fused per link per control step (the latest-arriving ones)


def init(value, phase, t0=0.0):
    dim = value.shape[0]
    return dict(arr=jnp.full(CAP, jnp.inf), sent=jnp.zeros(CAP), val=jnp.zeros((CAP, dim)),
                occ=jnp.zeros(CAP, dtype=bool), write=jnp.zeros((), jnp.int32),
                phase=jnp.asarray(phase, jnp.float32), out=value, out_sent=jnp.asarray(t0, jnp.float32),
                last=jnp.asarray(t0, jnp.float32))


def _window(t, start, dur):
    return (t >= start) & (t < start + dur)


def step(st, prm, t, value, rnd, blocked=False):
    """Advance one control step. `prm` keys: rate, d0, sigma_j, p_drop,
    freeze_t, freeze_dur, burst_t, burst_dur, burst_delay, burst_window,
    outage_t, outage_dur. `value` already carries measurement noise.
    `rnd` = (uniform for drop, unused, standard normal for jitter).
    Returns (state, out_value, valid, age, arrivals)."""
    phase = st['phase'] + prm['rate'] * dyn.DT
    send = phase >= 1.0
    phase = jnp.where(send, phase - 1.0, phase)
    outage = _window(t, prm['outage_t'], prm['outage_dur'])
    push = send & (rnd[0] >= prm['p_drop']) & ~outage & ~jnp.asarray(blocked)
    burst = _window(t, prm['burst_t'], prm['burst_dur'])
    sigma_j = jnp.where(burst & (prm['burst_window'] > 0),
                        prm['burst_window'] / prm['rate'], prm['sigma_j'])
    delay = prm['d0'] + jnp.where(burst, prm['burst_delay'], 0.0) + jnp.abs(rnd[2]) * sigma_j
    slot = st['write'] % CAP
    sel = (jnp.arange(CAP) == slot) & push
    arr = jnp.where(sel, t + delay, st['arr'])
    sent = jnp.where(sel, t, st['sent'])
    val = jnp.where(sel[:, None], value[None], st['val'])
    occ = st['occ'] | sel
    delivered = occ & (arr <= t + EPS_T)
    any_d = jnp.any(delivered)
    last_idx = jnp.argmax(jnp.where(delivered, arr, -jnp.inf))
    new_out = jnp.where(any_d, val[last_idx], st['out'])
    new_sent = jnp.where(any_d, sent[last_idx], st['out_sent'])
    last = jnp.where(any_d, t, st['last'])
    frozen = _window(t, prm['freeze_t'], prm['freeze_dur'])
    out = jnp.where(frozen, st['out'], new_out)
    out_sent = jnp.where(frozen, st['out_sent'], new_sent)
    valid = jnp.where(frozen, True, (t - last) <= T_VALID + EPS_T)
    new = dict(arr=jnp.where(delivered, jnp.inf, arr), sent=sent, val=val,
               occ=occ & ~delivered, write=st['write'] + push.astype(jnp.int32),
               phase=phase, out=out, out_sent=out_sent, last=last)
    # arrival stream: the MAX_ARRIVALS latest-arriving delivered packets, oldest arrival first
    key_arr = jnp.where(delivered, arr, -jnp.inf)
    top, idx = jax.lax.top_k(key_arr, MAX_ARRIVALS)
    idx, top = idx[::-1], top[::-1]
    a_mask = jnp.isfinite(top)
    a_val = jnp.where(frozen, st['out'][None], val[idx])
    arrivals = dict(val=a_val, arr=jnp.where(a_mask, top, 0.0), sent=sent[idx], mask=a_mask.astype(jnp.float32))
    return new, out, valid.astype(jnp.float32), t - out_sent, arrivals
