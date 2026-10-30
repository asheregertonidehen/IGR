"""Evaluation runner.

All M models of a set are vmapped over the SAME episodes and the SAME per-step noise keys
(paired comparisons; episodes depend only on eval seed + suite). Deterministic actions:
tanh(mean) (clip for B0). No auto-reset: a crashed episode is frozen and counted as a failure.
Nested scan: `stride` inner steps per outer step so probe hidden states can be subsampled.

A ModelSet runs one static estimator mode (`est`, sensing/layer.EST_MODES); per-model estimator
tuning comes from the model configs. `make_pair_fn` runs every episode twice in lock-step, with
and without a scripted event (paired counterfactual), and returns per-step comparisons.
"""
from dataclasses import dataclass, field
import zlib

import jax
import jax.numpy as jnp
import numpy as np

from models import inputs as I
from sim import env as envm
from sim import specs
from train import common


@dataclass
class ModelSet:
    names: list
    arch: object
    params: object          # pytree with leading model axis M
    cfg: dict               # dict of [M] float arrays
    out: str = 'tanh'       # 'tanh' | 'clip'
    meta: dict = field(default_factory=dict)
    est: str = None         # static estimator mode of this program: None | 'nots' | 'ts'

    @property
    def M(self):
        return len(self.names)


def suite_key(eval_seed, suite):
    return jax.random.fold_in(jax.random.PRNGKey(eval_seed), zlib.crc32(suite.encode()) & 0x7FFFFFFF)


def event_time(ep):
    """Earliest scheduled event in an episode (for time-to-recover); 1e9 if none."""
    ev, s = ep['events'], ep['sens']
    cands = [jnp.where(ev['gust_on'] > 0, ev['gust_t'], 1e9),
             jnp.where(jnp.abs(ev['pay_dm']) > 0, ev['pay_t'], 1e9),
             jnp.where(ev['loss_k'] < 1, ev['loss_t'], 1e9),
             s['vel']['freeze_t'], s['pos']['freeze_t'], s['vel']['burst_t'], s['pos']['burst_t'],
             s['ref']['burst_t'], s['vel']['outage_t'], s['act_t'], ep['script_t'],
             jnp.where((s['flow']['dip_on'] > 0) & (s['flow']['is_flow'] > 0), s['flow']['dip_t'], 1e9)]
    return jnp.min(jnp.stack(cands))


def _est_env(env, cfg_m):
    if 'ekf_q' not in cfg_m:
        return env
    return envm.with_est(env, dict(q=cfg_m['ekf_q'], dhat=cfg_m['ekf_dhat'], tau=cfg_m['leak_tau']))


def _sv_inject(arch, cfg, ch, dv):
    """Add a normalized ext-velocity perturbation dv (3,) to the features (for S_v)."""
    if arch.kind == 'gated':
        return dict(ch, ext_vel=ch['ext_vel'].at[:3].add(dv))
    k = cfg['input_kind']
    x = ch['x']
    for kind, sl in I.VEL_SLICE.items():
        x = x.at[sl].add(jnp.where(k == kind, 1.0, 0.0) * dv)
    return dict(ch, x=x)


def _heads_phys(aux, shape):
    """Physical-scale head outputs: velocity error (m/s), external force (/m g), effective T/W."""
    if 'truth_head' in aux:
        th = I.truth_destandardize(aux['truth_head'])
        idh = I.destandardize(aux['id_head'], 'id')
        return th[..., I.TRUTH_SLICES['v_err']] * I.VEL, th[..., I.TRUTH_SLICES['force']], idh[..., 0]
    z = jnp.zeros(shape + (3,))
    return z, z, jnp.zeros(shape)


def _model_fns(ms, est, perturb):
    arch, out = ms.arch, ms.out

    def act(mean):
        return jnp.clip(mean, -1, 1) if out == 'clip' else jnp.tanh(mean)

    def model_step(p, cfg, carry, o, pv, pert):
        if perturb == 'raw_noise':
            o = dict(o, ext_pos=o['ext_pos'] + pert['raw_pos'], ext_vel=o['ext_vel'] + pert['raw_vel'])
        ch = arch.features(o, pv, cfg)
        if perturb in ('r_noise', 'r_mean'):
            ch = dict(ch, r_a=pert['r_a'], r_b=pert['r_b'])
        carry, mean, aux = arch.step(p, cfg, carry, ch)
        return carry, act(mean), aux, ch

    def sens_fn(p, cfg, carry, ch):
        f = lambda dv: act(arch.step(p, cfg, carry, _sv_inject(arch, cfg, ch, dv))[1])
        J = jax.jacfwd(f)(jnp.zeros(3))
        return jnp.sqrt(jnp.sum(J * J))
    return act, model_step, sens_fn


def make_eval_fn(ms, n_eps, steps, stride=10, record=(), est=None, perturb=None):
    """Compile an evaluation. record subset of {'traj', 'probe', 'discrim', 'est', 'rel'}; perturb (9.2b)
    None | 'r_noise' | 'r_mean' | 'raw_noise'. Returns f(params, cfg, pool, key, pert) -> (rec, orec)."""
    arch = ms.arch
    n_outer = steps // stride
    _, model_step, sens_fn = _model_fns(ms, est, perturb)
    vm = lambda f: jax.vmap(jax.vmap(f, (None, None, 0, 0, 0, 0)), (0, 0, 0, 0, 0, 0))
    estep = jax.vmap(jax.vmap(lambda s, a, k: envm.step(s, a, k, est)), (0, 0, None))

    def run(params, cfg, pool, key, pert):
        env0 = jax.vmap(lambda e: envm.reset(e, est))(pool)
        env = jax.tree.map(lambda x: jnp.broadcast_to(x, (ms.M,) + x.shape), env0)
        env = jax.vmap(lambda e, c: jax.vmap(lambda ee: _est_env(ee, c))(e))(env, cfg)
        c0 = arch.init_carry()
        carry = jax.tree.map(lambda x: jnp.broadcast_to(x, (ms.M, n_eps) + x.shape), c0)

        def inner(c, k):
            env, carry, t = c
            o = env['obs']
            pv = jax.vmap(jax.vmap(envm.privileged))(env)
            carry_prev = carry
            kp = jax.random.fold_in(k, 999)
            pt = _pert_step(pert, perturb, kp, t, ms.M, n_eps)
            carry, a, aux, ch = vm(model_step)(params, cfg, carry, o, pv, pt)
            keys = jax.random.split(k, n_eps)
            env2, _, _ = estep(env, a, keys)
            rec = dict(perr=jnp.sqrt(jnp.sum(pv['p_err'] ** 2, -1)), zerr=pv['p_err'][..., 2], alive=env['alive'])
            if 'traj' in record:
                rec.update(pos=env['phys']['p'], quat=env['phys']['q'], p_ref=env['phys']['p'] - pv['p_err'],
                           action=a, rpm=env['phys']['rpm'], v=env['phys']['v'])
            if 'discrim' in record:
                sv = jax.vmap(jax.vmap(sens_fn, (None, None, 0, 0)), (0, 0, 0, 0))(params, cfg, carry_prev, ch) \
                    if arch.kind != 'b0' else rec['perr'] * 0
                vel_h, force_h, t2w_h = _heads_phys(aux, a.shape[:2])
                rec.update(S_v=sv, gates=aux.get('gates', jnp.zeros(a.shape[:2] + (3,))),
                           nis=aux.get('nis', jnp.zeros(a.shape[:2] + (3,))),
                           force_head=jnp.linalg.norm(force_h, axis=-1), t2w_head=t2w_h,
                           vel_head_err=jnp.linalg.norm(vel_h - pv['v_err'], axis=-1))
            if 'est' in record:
                rec.update(est_perr=jnp.linalg.norm(o['est_p'] - pv['p_err'], axis=-1),
                           est_verr=jnp.linalg.norm(o['est_v'] - pv['v_err'], axis=-1),
                           leak_verr=jnp.linalg.norm(o['leaky_v'] - pv['v_err'], axis=-1),
                           raw_verr=jnp.linalg.norm(o['ext_vel'] - pv['v_err'], axis=-1),
                           raw_perr=jnp.linalg.norm(o['ext_pos'] - pv['p_err'], axis=-1))
            return (env2, carry, t + 1), rec

        def outer(c, k):
            c, rec = jax.lax.scan(inner, c, jax.random.split(k, stride))
            orec = {}
            if 'probe' in record or 'rel' in record:
                env, carry, _ = c
                pv = jax.vmap(jax.vmap(envm.privileged))(env)
                hid = jnp.concatenate(jax.tree.leaves(carry), axis=-1)
                if 'probe' in record:
                    d_act = jnp.where(env['t'] >= env['ep']['sens']['act_t'], env['ep']['sens']['act_d1'],
                                      env['ep']['sens']['act_d0'])
                    orec = dict(hidden=hid, v_true=pv['v_err'], v_deliv=env['obs']['ext_vel'],
                                t2w=pv['identity'][..., 0], f_ext=pv['ext'][..., :3] / (pv['mass'][..., None] * I.ACC),
                                act_delay=d_act, alive=env['alive'])
                if 'rel' in record:
                    orec.update(r=carry['r'] if 'r' in carry else hid[..., :1], alive=env['alive'],
                                ext_pos=env['obs']['ext_pos'], ext_vel=env['obs']['ext_vel'])
            return c, (rec, orec)

        _, (rec, orec) = jax.lax.scan(outer, (env, carry, jnp.zeros((), jnp.int32)), jax.random.split(key, n_outer))
        # [n_outer, stride, M, n] -> [M, n, steps]
        rec = jax.tree.map(lambda x: jnp.moveaxis(x.reshape((n_outer * stride,) + x.shape[2:]), 0, 2), rec)
        orec = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 2), orec)
        return rec, orec

    return jax.jit(run)


def _pert_step(pert, perturb, key, t, M, n):
    """Per-step perturbation arrays [M, n, ...] for the reliance test."""
    if perturb is None:
        return jnp.zeros((M, n))
    if perturb == 'r_noise':
        R = pert['r_std'].shape[-1]
        z = jax.random.normal(key, (M, n, R))
        return dict(r_a=jnp.ones((M, n, 1)), r_b=0.5 * pert['r_std'][:, None, :] * z)
    if perturb == 'r_mean':
        return dict(r_a=jnp.zeros((M, n, 1)), r_b=pert['r_mean'])
    k1, k2 = jax.random.split(key)
    return dict(raw_pos=0.5 * pert['pos_std'][:, None, :] * jax.random.normal(k1, (M, n, 3)),
                raw_vel=0.5 * pert['vel_std'][:, None, :] * jax.random.normal(k2, (M, n, 3)))


_CACHE = {}
DEFAULT_STEPS = None  # profile override of the episode length (smoke only)


def _pool_meta(pl):
    return dict(event_t=np.asarray(jax.vmap(event_time)(pl)), drone_idx=np.asarray(pl['drone_idx']),
                term_pos=np.asarray(pl['term_pos']), task=np.asarray(pl['task']['kind']),
                cat=np.asarray(pl['cat']), t2w=np.asarray(pl['drone']['t2w']))


def evaluate(ms, suite, n_eps, eval_seed=54321, chunk=256, stride=10, record=(), spec_extra=None,
             spec=None, meta=None, pool=None, steps=None, perturb=None, pert=None, key=None):
    """Run a model set on a suite; returns dict of numpy arrays [M, n_eps, ...]."""
    if spec is None:
        spec, meta = specs.suite_spec(suite, spec_extra)
    dset = common.drone_set(meta['drones'])
    steps = steps or DEFAULT_STEPS or int(round(float(spec['episode_s']) / 0.01))
    steps = max(stride, steps // stride * stride)
    chunk = min(chunk, n_eps)
    key = suite_key(eval_seed, suite) if key is None else key
    sig = (id(ms.arch), ms.M, chunk, steps, stride, tuple(record), ms.out, ms.est, perturb)
    if sig not in _CACHE:
        _CACHE[sig] = make_eval_fn(ms, chunk, steps, stride, record, ms.est, perturb)
    fn = _CACHE[sig]
    outs, metas = [], []
    for start in range(0, n_eps, chunk):
        kc = jax.random.fold_in(key, start)
        if pool is None:
            pl = common.sample_pool(jax.random.fold_in(kc, 7), dset, spec, 1.0, chunk)
        else:
            pl = jax.tree.map(lambda x: x[start:start + chunk], pool)
        pc = None
        if perturb is not None:
            pc = {k: (v[:, start:start + chunk] if k == 'r_mean' else v) for k, v in pert.items()}
        rec, orec = fn(ms.params, ms.cfg, pl, kc, pc if pc is not None else jnp.zeros(()))
        outs.append(jax.tree.map(np.asarray, (rec, orec)))
        metas.append(_pool_meta(pl))
    rec = jax.tree.map(lambda *x: np.concatenate(x, axis=1), *[o[0] for o in outs])
    orec = jax.tree.map(lambda *x: np.concatenate(x, axis=1), *[o[1] for o in outs]) if outs[0][1] else {}
    ep_meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    return dict(rec=rec, probe=orec, meta=ep_meta, names=list(ms.names), suite=suite)


# ------------------------------------------------------------------ paired counterfactuals
def make_pair_fn(ms, n_eps, steps):
    """Event and counterfactual episodes run in lock-step with identical noise."""
    arch = ms.arch
    est = ms.est
    _, model_step, sens_fn = _model_fns(ms, est, None)
    vm = jax.vmap(jax.vmap(jax.vmap(model_step, (None, None, 0, 0, 0, None)), (None, None, 0, 0, 0, None)),
                  (0, 0, 0, 0, 0, None))     # [M, 2, n]
    estep = jax.vmap(jax.vmap(jax.vmap(lambda s, a, k: envm.step(s, a, k, est), (0, 0, 0)), (0, 0, None)),
                     (0, 0, None))

    def run(params, cfg, pool2, key):
        env0 = jax.vmap(jax.vmap(lambda e: envm.reset(e, est)))(pool2)            # [2, n]
        env = jax.tree.map(lambda x: jnp.broadcast_to(x, (ms.M,) + x.shape), env0)
        env = jax.vmap(lambda e, c: jax.vmap(jax.vmap(lambda ee: _est_env(ee, c)))(e))(env, cfg)
        carry = jax.tree.map(lambda x: jnp.broadcast_to(x, (ms.M, 2, n_eps) + x.shape), arch.init_carry())

        def step(c, k):
            env, carry = c
            pv = jax.vmap(jax.vmap(jax.vmap(envm.privileged)))(env)
            carry_prev = carry
            carry, a, aux, ch = vm(params, cfg, carry, env['obs'], pv, jnp.zeros(()))
            env2, _, _ = estep(env, a, jax.random.split(k, n_eps))
            h = jnp.concatenate(jax.tree.leaves(carry), axis=-1)
            perr = jnp.sqrt(jnp.sum(pv['p_err'] ** 2, -1))
            rec = dict(perr_ev=perr[:, 0], perr_cf=perr[:, 1], alive_ev=env['alive'][:, 0], alive_cf=env['alive'][:, 1],
                       adev=jnp.linalg.norm(a[:, 0] - a[:, 1], axis=-1),
                       mcont=jnp.linalg.norm(h[:, 0] - h[:, 1], axis=-1) / jnp.maximum(jnp.linalg.norm(h[:, 1], axis=-1), 1e-6))
            if arch.kind == 'gated':
                hr0, hr1 = carry['rep'][:, 0], carry['rep'][:, 1]
                rec['mcont_rep'] = jnp.linalg.norm(hr0 - hr1, axis=-1) / jnp.maximum(jnp.linalg.norm(hr1, axis=-1), 1e-6)
            ev = lambda x: x[:, 0]
            aux_ev = jax.tree.map(ev, aux)
            vel_h, force_h, t2w_h = _heads_phys(aux_ev, a.shape[:1] + a.shape[2:3])
            sv = jax.vmap(jax.vmap(sens_fn, (None, None, 0, 0)), (0, 0, 0, 0))(
                params, cfg, jax.tree.map(ev, carry_prev), jax.tree.map(ev, ch)) if arch.kind != 'b0' else rec['adev'] * 0
            rec.update(gates=aux_ev.get('gates', jnp.zeros(a.shape[:1] + a.shape[2:3] + (3,))),
                       nis=aux_ev.get('nis', jnp.zeros(a.shape[:1] + a.shape[2:3] + (3,))),
                       force_head=jnp.linalg.norm(force_h, axis=-1), t2w_head=t2w_h, S_v=sv,
                       vel_head_err=jnp.linalg.norm(vel_h - pv['v_err'][:, 0], axis=-1))
            return (env2, carry), rec

        _, rec = jax.lax.scan(step, (env, carry), jax.random.split(key, steps))
        return jax.tree.map(lambda x: jnp.moveaxis(x, 0, 2), rec)       # [M, n, steps, ...]
    return jax.jit(run)


def evaluate_pair(ms, pool_ev, pool_cf, key, steps):
    n = int(pool_ev['drone_idx'].shape[0])
    sig = ('pair', id(ms.arch), ms.M, n, steps, ms.out, ms.est)
    if sig not in _CACHE:
        _CACHE[sig] = make_pair_fn(ms, n, steps)
    pool2 = jax.tree.map(lambda a, b: jnp.stack([a, b]), pool_ev, pool_cf)
    return jax.tree.map(np.asarray, _CACHE[sig](ms.params, ms.cfg, pool2, key))


# ----------------------------------------------------------------------- metrics
DT = 0.01
SKIP_S = 2.0       # initial-state transient excluded from RMSE / oscillation metrics
SURVIVAL_T = 5.0   # event-conditioned survival: alive at 5 s -> alive at the end


def skip_steps(T):
    """Initial-state transient excluded from RMSE / PSD / estimator metrics: 2 s, or T/4 for the
    short smoke episodes (full-size episodes are 2000 steps, so always 2 s there)."""
    return min(int(round(SKIP_S / DT)), T // 4)


def episode_metrics(perr, zerr, alive, event_t, term_pos):
    """perr/zerr/alive [n, T] for one model -> dict of per-episode arrays."""
    n, T = perr.shape
    s0 = skip_steps(T)
    success = alive[:, -1].astype(bool)
    crashed_before = ~alive.astype(bool)
    thr = term_pos[:, None]
    clipped = np.where(crashed_before, thr, np.minimum(perr, thr))
    rmse_all = np.sqrt(np.mean(clipped[:, s0:] ** 2, axis=1))
    rmse = np.where(success, np.sqrt(np.mean(perr[:, s0:] ** 2, axis=1)), np.nan)
    max_err = np.where(success, perr[:, s0:].max(axis=1), np.nan)
    z = zerr[:, s0:] - zerr[:, s0:].mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(z, axis=1)) ** 2 * 2 / z.shape[1] ** 2
    freqs = np.fft.rfftfreq(z.shape[1], DT)
    band = (freqs >= 0.5) & (freqs <= 5.0)
    zpow = np.where(success, spec[:, band].sum(1), np.nan)
    zfrac = np.where(success, spec[:, band].sum(1) / np.maximum(spec[:, 1:].sum(1), 1e-12), np.nan)
    rec = np.full(n, np.nan)
    hold = int(0.5 / DT)
    for i in np.where(success & (event_t < T * DT - 1.0) & (event_t > 2.5))[0]:
        k = int(event_t[i] / DT)
        pre = perr[i, k - 200:k].mean()
        ok = perr[i, k + 10:] <= 1.5 * max(pre, 0.02)
        run = np.convolve(ok.astype(float), np.ones(hold), 'valid') >= hold
        idx = np.argmax(run) if run.any() else -1
        rec[i] = (idx + 10) * DT if idx >= 0 else np.inf
    k5 = min(int(SURVIVAL_T / DT), T) - 1
    return dict(success=success.astype(float), rmse=rmse, rmse_clipped=rmse_all, max_err=max_err,
                zosc_power=zpow, zosc_frac=zfrac, recover_s=rec, alive5=alive[:, k5].astype(float))


def metrics_table(result):
    """Long-format rows: one per (model, episode)."""
    import pandas as pd
    rows = []
    rec, meta = result['rec'], result['meta']
    for m, name in enumerate(result['names']):
        em = episode_metrics(rec['perr'][m], rec['zerr'][m], rec['alive'][m], meta['event_t'], meta['term_pos'])
        df = pd.DataFrame(em)
        df['episode'] = np.arange(len(df))
        df['model'] = name
        df['suite'] = result['suite']
        df['drone_idx'] = meta['drone_idx']
        df['task'] = meta['task']
        df['t2w'] = meta['t2w']
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
