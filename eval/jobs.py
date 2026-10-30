"""Evaluation and selection jobs.

Groups: 'B0', 'E', 'T' or a student family ('F1_S1', 'F1_S2', 'F2b', ...). Suite results go to RESULTS/eval/<group>/<suite>.parquet (one row per
model x episode), written per suite so interrupted evaluations resume at the next missing suite.
Every selection (checkpoints, lambda_s*, classical tuning) uses val_id with the selection seed
only; reporting evaluations use the eval seed.
"""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from eval import b0_harness, modelsets, runner
from models import inputs as I
from models import variants as V
from sim import env as envm
from sim import specs, tasks
from train import common

EVAL_DIR = common.RESULTS / 'eval'
EVAL_GROUPS = ('id', 'test_id', 'clean', 'heldout', 'sweep', 'descriptive')


def load_group(group):
    """List of ModelSets (one per static estimator mode)."""
    if group == 'B0':
        return [b0_harness.b0_modelset(truth=False)]
    if group == 'E':
        return [modelsets.load_experts()]
    if group in V.TEACHER_PROGRAMS:
        return [modelsets.load_teachers(group)]
    return modelsets.load_family(group)


def _concat_tables(sets, fn):
    return pd.concat([fn(ms) for ms in sets], ignore_index=True)


def eval_group(ctx, group):
    sets = load_group(group)
    out = EVAL_DIR / group
    out.mkdir(parents=True, exist_ok=True)
    prof = ctx.profile
    n = prof['episodes_per_suite']
    done = []
    for suite in specs.suite_names(EVAL_GROUPS):
        path = out / f'{suite}.parquet'
        if path.exists():
            done.append(suite)
            continue
        frames, psd_rows = [], []
        for ms in sets:
            res = runner.evaluate(ms, suite, n, eval_seed=prof['eval_seed'], chunk=prof['eval_chunk'])
            frames.append(runner.metrics_table(res))
            if suite == 'delay_100':   # z-error PSD for fig. 8 (successful episodes, after 2 s)
                z = res['rec']['zerr'][:, :, runner.skip_steps(res['rec']['zerr'].shape[-1]):]
                ok = res['rec']['alive'][:, :, -1].astype(bool)
                z = z - z.mean(-1, keepdims=True)
                psd = np.abs(np.fft.rfft(z, axis=-1)) ** 2 * 2 * runner.DT / z.shape[-1]
                for m in range(psd.shape[0]):
                    psd_rows.append((res['names'][m], psd[m][ok[m]].mean(0) if ok[m].any() else np.full(psd.shape[-1], np.nan)))
                freqs = np.fft.rfftfreq(z.shape[-1], runner.DT)
        df = pd.concat(frames, ignore_index=True)
        if psd_rows:
            np.savez(out / 'psd_delay_100.npz', psd=np.stack([p for _, p in psd_rows]), freqs=freqs,
                     names=np.array([nm for nm, _ in psd_rows]))
        df.to_parquet(path)
        done.append(suite)
        ctx.log(f'{group}/{suite}: success ' + ', '.join(
            f'{m}={g.success.mean():.2f}' for m, g in list(df.groupby('model'))[:4]))
    return dict(done=done, episodes=n)


# ------------------------------------------------------------ checkpoint selection
def _sel_eval(ms_list, prof):
    out = {}
    for ms in ms_list:
        df = runner.metrics_table(runner.evaluate(ms, 'val_id', prof['selection_episodes'],
                                                  eval_seed=prof['selection_seed'], chunk=prof['eval_chunk']))
        for m, g in df.groupby('model'):
            out[m] = [float(g.success.mean()), float(np.nanmean(g.rmse)) if g.rmse.notna().any() else float('inf')]
    return out


def _better(a, b):
    """Selection rule: higher val_id success, ties broken by lower RMSE."""
    return b is None or (a[0], -a[1]) > (b[0], -b[1])


def select_family(ctx, fam, job_dir=None):
    """Evaluate every selection checkpoint of a family on val_id (selection seed) and keep the
    best per model. Evaluations are cached, so re-selection after an extension only
    evaluates the new checkpoints."""
    job_dir = job_dir or ctx.job_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    fdir = common.RESULTS / fam
    _, fmeta = common.load_params(fdir / 'params.pkl')
    cache_p = job_dir / 'evals.json'
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    store = common.SelectionStore(fdir)
    lam, tuning = modelsets.lambda_star(), modelsets.est_tuning()
    best = {}
    for prog in fmeta['programs']:
        tag = prog['tag']
        idx = [i for i, m in enumerate(fmeta['models']) if m[0] in prog['models']]
        models = [tuple(fmeta['models'][i]) for i in idx]
        cache.setdefault(tag, {})
        for ep in store.steps(tag):
            if str(ep) not in cache[tag]:
                p = store.load(tag, ep)
                sets = modelsets.family_sets(fam, p, models, lam=fmeta.get('lambda_star') or lam, tuning=tuning)
                cache[tag][str(ep)] = _sel_eval(sets, ctx.profile)
                common.atomic_write_json(cache_p, cache)
                ctx.log(f'select {tag} epoch {ep}: ' + ', '.join(f'{k}={v[0]:.3f}' for k, v in list(cache[tag][str(ep)].items())[:3]))
            for j, m in enumerate(models):
                r = cache[tag][str(ep)][m[0]]
                cur = best.get(m[0])
                if _better(r, None if cur is None else (cur[1], cur[2])):
                    best[m[0]] = (tag, r[0], r[1], ep, j)
    names = [m[0] for m in fmeta['models']]
    chosen = []
    for n in names:
        tag, _, _, ep, j = best[n]
        chosen.append(jax.tree.map(lambda x: x[j], store.load(tag, ep)))
    stacked = jax.tree.map(lambda *x: jnp.stack(x), *chosen)
    curves = {n: {ep: cache[best[n][0]][ep][n] for ep in sorted(cache[best[n][0]], key=int)} for n in names}
    meta = dict(fmeta, selected={n: dict(epoch=int(best[n][3]), success=best[n][1], rmse=best[n][2]) for n in names},
                curves=curves, rule='max val_id success (selection seed), ties -> lower RMSE')
    common.save_params(job_dir / 'params.pkl', stacked, meta)
    common.atomic_write_json(job_dir / 'meta.json', {k: v for k, v in meta.items() if k != 'est_tuning'})
    return dict(selected={n: meta['selected'][n]['epoch'] for n in names})


def select_lambda(ctx):
    """lambda_s* = grid value with the highest val_id success of the selected F1_S1 checkpoints,
    averaged over teacher seeds; ties -> lower RMSE. Never extended after results."""
    meta = json.loads((common.RESULTS / 'select_ckpt_F1_S1' / 'meta.json').read_text())
    rows = []
    for lam in V.LAMBDA_GRID:
        v = V.lam_name(lam)
        sel = [s for n, s in meta['selected'].items() if V.parse_model(n)[0] == v]
        if sel:
            rows.append(dict(variant=v, lambda_s=lam, success=float(np.mean([s['success'] for s in sel])),
                             rmse=float(np.mean([s['rmse'] for s in sel])), n_teacher_seeds=len(sel)))
    best = sorted(rows, key=lambda r: (-r['success'], r['rmse']))[0]
    sel = dict(lambda_star=best['lambda_s'], variant=best['variant'],
               rule='max mean val_id success of the selected checkpoints over teacher seeds, tie -> lower RMSE',
               at_grid_edge=bool(best['lambda_s'] in (min(V.LAMBDA_GRID), max(V.LAMBDA_GRID))), sweep=rows)
    common.atomic_write_json(common.RESULTS / 'selection.json', sel)
    ctx.log(f"lambda_s* on val_id: {best['lambda_s']}" + (' (at the grid edge; reported, grid not extended)'
                                                          if sel['at_grid_edge'] else ''))
    return sel


# ------------------------------------------------------------ convergence check
def plateau(curve, n_epochs, cfg):
    """curve {epoch: [success, rmse]} -> (converged, detail). Best checkpoint in the last
    window_frac of the epochs vs the best before it."""
    eps = sorted(int(e) for e in curve)
    cut = (1 - cfg['window_frac']) * n_epochs
    pick = lambda es: max(((curve[str(e)][0], -curve[str(e)][1]) for e in es), default=None)
    late, early = pick([e for e in eps if e > cut]), pick([e for e in eps if e <= cut])
    if late is None or early is None:
        return True, dict(reason='too few checkpoints')
    d_succ = late[0] - early[0]
    d_rmse = (-early[1] - -late[1]) / max(-early[1], 1e-9) if np.isfinite(early[1]) and np.isfinite(late[1]) else 0.0
    ok = not (d_succ > cfg['success_pp'] / 100 or d_rmse > cfg['rmse_pct'] / 100)
    return ok, dict(d_success=float(d_succ), rmse_improvement=float(d_rmse))


def variant_curves(fams):
    """{variant: {epoch: [success, rmse]}}: val_id curves averaged over all students of a variant.
    The plateau test is per VARIANT; testing each of ~100 noisy
    single-student curves would trigger an extension by chance alone."""
    acc = {}
    for fam in fams:
        m = json.loads((common.RESULTS / f'select_ckpt_{fam}' / 'meta.json').read_text())
        for n, curve in m['curves'].items():
            for ep, (s, r) in curve.items():
                acc.setdefault(V.parse_model(n)[0], {}).setdefault(ep, []).append((s, r))
    return {v: {ep: [float(np.mean([x[0] for x in xs])),
                     float(np.mean([x[1] for x in xs if np.isfinite(x[1])] or [np.inf]))]
                for ep, xs in eps.items()} for v, eps in acc.items()}


def converge(ctx, stage):
    """Plateau test over the stage's students; if any fails, extend ALL students (stage A: the
    stage-A families; stage B: every family, to keep one budget) by 50 %, at most twice."""
    from orchestrator.context import JobContext
    from train import distill
    cfg = ctx.profile['convergence']
    st = distill.convergence_state()
    while True:
        n_epochs = distill.target_epochs(ctx.budget)
        fails = {}
        for v, curve in variant_curves(V.STAGES[stage]).items():
            ok, det = plateau(curve, n_epochs, cfg)
            if not ok:
                fails[v] = det
        st['history'].append(dict(stage=stage, epochs=n_epochs, not_converged=fails))
        if not fails or st['n_ext'] >= cfg['max_extensions']:
            if fails:
                ctx.note(f'stage {stage}: {len(fails)} students still not converged after {st["n_ext"]} extensions')
            common.atomic_write_json(common.RESULTS / 'convergence.json', st)
            return dict(n_ext=st['n_ext'], not_converged=list(fails), epochs=n_epochs)
        st['n_ext'] += 1
        common.atomic_write_json(common.RESULTS / 'convergence.json', st)
        ctx.note(f'stage {stage}: {len(fails)} students not converged ({list(fails)[:5]}...); extending ALL '
                 f'students to {distill.target_epochs(ctx.budget)} epochs (extension {st["n_ext"]})')
        fams = V.STAGES['A'] + (V.STAGES['B'] if stage == 'B' else [])
        for fam in fams:
            if not (common.RESULTS / fam / 'params.pkl').exists():
                continue
            c2 = JobContext(job=fam, job_dir=common.RESULTS / fam, profile=ctx.profile, budget=ctx.budget,
                            log_fn=ctx.log_fn)
            distill.run(c2, fam)
            select_family(ctx, fam, job_dir=common.RESULTS / f'select_ckpt_{fam}')


# ------------------------------------------------------------ classical estimator tuning
def tune_classical(ctx):
    """Select the ekf_nots (process-noise scale x nominal delay), ekf_ts (scale) and leaky-velocity
    (tau) settings on val_id (selection seed) by estimation accuracy while E_0 flies."""
    prof = ctx.profile
    sw = prof['classical_sweeps']
    e_all, e_meta = common.load_params(common.RESULTS / 'E' / 'params.pkl')
    e0 = jax.tree.map(lambda x: x[0], e_all)
    n = prof['selection_episodes']
    rows, sel = [], {}
    grids = {}
    if 'ekf_nots' in sw:
        grids['ekf_nots'] = [dict(q=q, dhat=d / 1000.0) for q in sw['ekf_nots']['process_noise_scale']
                             for d in sw['ekf_nots']['nominal_delay_ms']]
    if 'ekf_ts' in sw:
        grids['ekf_ts'] = [dict(q=q, dhat=0.0) for q in sw['ekf_ts']['process_noise_scale']]
    if 'leaky_vel' in sw:
        grids['leaky_vel'] = [dict(q=1.0, dhat=0.0, tau=t) for t in sw['leaky_vel']['tau_s']]
    est_of = dict(ekf_nots='nots', ekf_ts='ts', leaky_vel=None)
    for name, grid in grids.items():
        C = len(grid)
        cfg = {'input_kind': jnp.zeros(C), 'ekf_q': jnp.asarray([g['q'] for g in grid], jnp.float32),
               'ekf_dhat': jnp.asarray([g['dhat'] for g in grid], jnp.float32),
               'leak_tau': jnp.asarray([g.get('tau', 0.05) for g in grid], jnp.float32)}
        ms = runner.ModelSet(names=[f'{name}_{i}' for i in range(C)], arch=modelsets.EXPERT_CLEAN,
                             params=jax.tree.map(lambda x: jnp.broadcast_to(x, (C,) + x.shape), e0), cfg=cfg,
                             est=est_of[name])
        res = runner.evaluate(ms, 'val_id', n, eval_seed=prof['selection_seed'], chunk=prof['eval_chunk'], record=('est',))
        rec = res['rec']
        s0 = runner.skip_steps(rec['alive'].shape[-1])
        alive = rec['alive'][:, :, s0:].astype(bool)
        rmse = lambda k: [float(np.sqrt(np.mean(rec[k][c][:, s0:][alive[c]] ** 2))) if alive[c].any() else np.nan
                          for c in range(C)]
        if name == 'leaky_vel':
            v = rmse('leak_verr')
            score = [x / I.VEL for x in v]
            p = [np.nan] * C
        else:
            p, v = rmse('est_perr'), rmse('est_verr')
            score = [a / I.POS + b / I.VEL for a, b in zip(p, v)]
        rraw = lambda k: float(np.sqrt(np.mean(rec[k][0][:, s0:][alive[0]] ** 2))) if alive[0].any() else np.nan
        raw = dict(p=rraw('raw_perr'), v=rraw('raw_verr'))
        for c, g in enumerate(grid):
            rows.append(dict(estimator=name, **g, p_rmse=p[c], v_rmse=v[c], score=score[c], raw_p_rmse=raw['p'],
                             raw_v_rmse=raw['v']))
        if np.isfinite(score).any():
            i = int(np.nanargmin(score))
        else:   # no surviving steps (only possible with an untrained smoke expert): middle of the grid
            i = len(grid) // 2
            ctx.note(f'tune {name}: no finite score; using the grid midpoint {grid[i]}')
        raw_score = raw['p'] / I.POS + raw['v'] / I.VEL if name != 'leaky_vel' else raw['v'] / I.VEL
        # guard: a classical estimator that is much worse than the raw delivered data is mis-tuned/broken,
        # not a fair baseline; flag it loudly (the report and the job notes show it).
        sel[name] = dict(grid[i], score=score[i], raw_score=raw_score, worse_than_raw=bool(score[i] > 2.0 * raw_score))
        if sel[name]['worse_than_raw']:
            ctx.note(f'tune {name}: selected estimator error {score[i]:.3f} is > 2x the raw delivered data ({raw_score:.3f}); '
                     'the classical baseline is NOT trustworthy')
        ctx.log(f'tune {name}: selected {grid[i]} (score {score[i]:.4f}; raw delivered p {raw["p"]:.3f} m, v {raw["v"]:.3f} m/s)')
    out = dict(sel, sweep=rows, rule='min val_id estimation error (p/1 m + v/2 m/s; leaky: v/2 m/s) with E_0 flying')
    d = ctx.job_dir
    common.atomic_write_json(d / 'selection.json', out)
    return {k: v for k, v in out.items() if k != 'sweep'}


# ------------------------------------------------------------- paired counterfactual
def _event_pools(prot_name, kind, n, seed):
    spec, keep, prot = specs.discrimination(prot_name)
    dset = common.drone_set('test')
    key = jax.random.PRNGKey(seed + 77)
    keys = jax.random.split(key, n)
    if float(prot.get('event_at_peak', 0)) > 0:
        t_ev = jax.vmap(lambda k: tasks.peak_accel_time(envm.episode_task(k, spec), spec['event_t'][0],
                                                        spec['event_t'][1]))(keys)
    else:
        t_ev = jnp.full(n, spec['event_t'][0])
    idx = jnp.arange(n) % int(dset['mass'].shape[0])
    mk = lambda k_: common.sample_pool(key, dset, spec, 1.0, n, drone_idx=idx,
                                       script=dict(kind=jnp.full(n, k_, jnp.int32), t=t_ev))
    return mk(kind), mk(0), np.asarray(t_ev), spec, keep


DETECT_SUSTAIN = 10      # 0.1 s: a single noisy step above 2 std is not a detection


def _first_dead(alive):
    """Index of the first step where the episode is no longer alive (T if it never crashed)."""
    dead = ~alive.astype(bool)
    return np.where(dead.any(1), dead.argmax(1), alive.shape[1])


def h2_metrics(rec, m, t_ev, win, keep, term_pos, ev_name):
    DT = runner.DT
    n, T = rec['perr_ev'][m].shape
    k0 = np.round(t_ev / DT).astype(int)
    kw = np.minimum(k0 + int(round(win / DT)), T - 1)
    ke = np.minimum(kw + int(round(keep['post_window'] / DT)), T - 1)
    kp = int(round(keep['pre_window'] / DT))
    a_ev, a_cf = rec['alive_ev'][m].astype(bool), rec['alive_cf'][m].astype(bool)
    crash = _first_dead(a_ev)
    surv_cf = a_cf[np.arange(n), ke]
    pre_alive = a_ev[np.arange(n), np.maximum(k0 - 1, 0)] & a_cf[np.arange(n), np.maximum(k0 - 1, 0)]
    valid = surv_cf & pre_alive
    X, ad, mc, mcr, mc_post = [], [], [], [], []
    caused = 0
    for i in np.where(valid)[0]:
        sl = slice(k0[i], ke[i] + 1)
        e = rec['perr_ev'][m][i].copy()
        e[crash[i]:] = term_pos[i]           # a crash counts as maximal harm from the crash onward
        X.append(np.sum(np.maximum(0.0, e[sl] - rec['perr_cf'][m][i][sl])) * DT)
        caused += int(crash[i] <= ke[i])
        ad.append(np.sum(rec['adev'][m][i][sl]) * DT)
        mc.append(np.sum(rec['mcont'][m][i][k0[i]:kw[i] + 1]) * DT)
        mc_post.append(np.sum(rec['mcont'][m][i][kw[i] + 1:ke[i] + 1]) * DT)
        if 'mcont_rep' in rec:
            mcr.append(np.sum(rec['mcont_rep'][m][i][k0[i]:ke[i] + 1]) * DT)
    out = dict(n_valid=int(valid.sum()), X=float(np.mean(X)) if X else np.nan, event_crashes=caused,
               action_dev=float(np.mean(ad)) if ad else np.nan, mcont_event=float(np.mean(mc)) if mc else np.nan,
               mcont_post=float(np.mean(mc_post)) if mc_post else np.nan,
               mcont_rep=float(np.mean(mcr)) if mcr else np.nan)
    pre_i = [i for i in np.where(valid)[0] if k0[i] - kp >= 0]
    win_mean = lambda key, j=None: float(np.mean([(rec[key][m][i][k0[i]:kw[i] + 1] if j is None else
                                                   rec[key][m][i][k0[i]:kw[i] + 1, j]).mean() for i in pre_i])) if pre_i else np.nan
    pre_mean = lambda key, j=None: float(np.mean([(rec[key][m][i][k0[i] - kp:k0[i]] if j is None else
                                                   rec[key][m][i][k0[i] - kp:k0[i], j]).mean() for i in pre_i])) if pre_i else np.nan
    for j, c in enumerate(I.EXT_CHANNELS):
        out[f'gate_{c}_pre'], out[f'gate_{c}_event'] = pre_mean('gates', j), win_mean('gates', j)
        out[f'nis_{c}_pre'], out[f'nis_{c}_event'] = pre_mean('nis', j), win_mean('nis', j)
    out['S_v_pre'], out['S_v_event'] = pre_mean('S_v'), win_mean('S_v')
    out['vel_head_err'] = win_mean('vel_head_err')
    det = keep['detect'].get(ev_name)
    if det:
        head, d_s = det
        key = 'force_head' if head == 'force' else 't2w_head'
        kd = int(round(d_s / DT))
        hits, tot = 0, 0
        for i in pre_i:
            if not a_ev[i, min(k0[i] + kd, T - 1)]:
                continue
            pre = rec[key][m][i][k0[i] - kp:k0[i]]
            mu, sd = pre.mean(), max(pre.std(), 1e-4)
            post = rec[key][m][i][k0[i]:min(k0[i] + kd + DETECT_SUSTAIN, T)]
            dev = post - mu if head == 'force' else np.abs(post - mu)
            above = (dev >= 2 * sd).astype(float)
            # a rise counts if the head stays >= 2 pre-event std for DETECT_SUSTAIN steps, starting within d_s
            run = np.convolve(above, np.ones(DETECT_SUSTAIN), 'valid') >= DETECT_SUSTAIN
            tot += 1
            hits += int(run[:kd + 1].any())
        out['detect_frac'] = hits / tot if tot else np.nan
        out['detect_n'] = tot
    return out


def event_traces(rec, m, t_ev, valid_all=True):
    """Mean traces aligned on the event onset, [-2, +4] s."""
    DT = runner.DT
    k0 = np.round(t_ev / DT).astype(int)
    rel = np.arange(-200, 400)
    T = rec['perr_ev'][m].shape[1]
    out = {}
    for key in ('perr_ev', 'perr_cf', 'adev', 'mcont', 'S_v', 'force_head', 't2w_head'):
        x = rec[key][m]
        out[key] = np.nanmean(np.stack([x[i][np.clip(k0[i] + rel, 0, T - 1)] for i in range(len(k0))]), 0)
    for key in ('gates', 'nis'):
        x = rec[key][m]
        out[key] = np.nanmean(np.stack([x[i][np.clip(k0[i] + rel, 0, T - 1)] for i in range(len(k0))]), 0)
    out['t'] = rel * DT
    return out


def discrimination(ctx, group, protocols=None):
    sets = load_group(group)
    prof = ctx.profile
    ddir = common.RESULTS / 'discrimination'
    ddir.mkdir(parents=True, exist_ok=True)
    result = {}
    for prot in protocols or specs.discrimination_protocols():
        _, keep, _ = specs.discrimination(prot)
        n = int(prof.get('event_drones', keep['n_drones']))
        out = {}
        for ev_name, kind in keep['events'].items():
            pool_ev, pool_cf, t_ev, spec, _ = _event_pools(prot, kind, n, prof['eval_seed'])
            # simulate up to the end of the latest analysis window (+0.2 s); nothing after it is used
            steps = min(int(round(float(spec['episode_s']) / runner.DT)),
                        int(np.ceil((t_ev.max() + keep['event_window'][ev_name] + keep['post_window'] + 0.2) / runner.DT)))
            summ = {}
            for ms in sets:
                rec = runner.evaluate_pair(ms, pool_ev, pool_cf, runner.suite_key(prof['eval_seed'], f'h2_{prot}_{ev_name}'),
                                           steps)
                for m, name in enumerate(ms.names):
                    summ[name] = h2_metrics(rec, m, t_ev, keep['event_window'][ev_name], keep,
                                            np.asarray(pool_ev['term_pos']), ev_name)
                    np.savez_compressed(ddir / f'{group}_{prot}_{ev_name}_{name}.npz', **event_traces(rec, m, t_ev))
            out[ev_name] = summ
            ctx.log(f'events {group}/{prot}/{ev_name}: X ' + ', '.join(f'{k}={v["X"]:.3f}' for k, v in list(summ.items())[:4]))
        common.atomic_write_json(ddir / f'{group}_{prot}.json', out)
        result[prot] = list(out)
    return result


def counterfactual_identity(ms, n=4, steps=60, seed=0):
    """Diagnostic used by the tests: with no event in either run, M(t) must be exactly 0."""
    pool, pool_cf, _, _, _ = _event_pools('primary', 0, n, seed)
    rec = runner.evaluate_pair(ms, pool, pool_cf, jax.random.PRNGKey(seed), steps)
    return float(np.max(np.abs(rec['mcont']))), float(np.max(np.abs(rec['adev'])))


# ------------------------------------------------------------------ reliance
RELIANCE_VARIANTS = ('Full', 'A3', 'A1', 'A7', 'B4', 'B5m', 'T_0')


def reliance(ctx, group):
    sets = load_group(group)
    prof = ctx.profile
    n = prof['probe_episodes']
    out = {}
    lam = modelsets.lambda_star()
    keep_v = set(RELIANCE_VARIANTS) | ({V.lam_name(lam)} if lam is not None else set())
    for ms in sets:
        idx = [i for i, nm in enumerate(ms.names) if V.parse_model(nm)[0] in keep_v or nm in keep_v]
        if not idx:
            continue
        ms = modelsets.subset(ms, idx)
        gated = ms.arch.kind == 'gated'
        for suite in ('val_id', 'delay_100'):
            ev = lambda **kw: runner.evaluate(ms, suite, n, eval_seed=prof['eval_seed'] + 9, chunk=prof['eval_chunk'], **kw)
            base = ev(record=('rel',))
            pr = base['probe']
            alive = pr['alive'].astype(bool)          # [M, n, n_outer]
            pert = {}
            if gated:
                r = pr['r']
                r_std = np.stack([r[m][alive[m]].std(0) for m in range(ms.M)])
                r_mean = np.stack([np.stack([r[m, i][alive[m, i]].mean(0) if alive[m, i].any() else np.zeros(r.shape[-1])
                                             for i in range(n)]) for m in range(ms.M)])
                pert['r_noise'] = dict(r_std=jnp.asarray(r_std))
                pert['r_mean'] = dict(r_mean=jnp.asarray(r_mean))
            sd = lambda k: jnp.asarray(np.stack([pr[k][m][alive[m]].std(0) for m in range(ms.M)]))
            pert['raw_noise'] = dict(pos_std=sd('ext_pos'), vel_std=sd('ext_vel'))
            tabs = {'none': runner.metrics_table(base)}
            for kind, pt in pert.items():
                tabs[kind] = runner.metrics_table(ev(perturb=kind, pert=pt))
            for nm in ms.names:
                row = {}
                for kind, df in tabs.items():
                    d = df[df.model == nm]
                    row[kind] = dict(success=float(d.success.mean()), rmse=float(np.nanmean(d.rmse)) if d.rmse.notna().any() else np.nan)
                for kind in pert:
                    row[f'd_success_{kind}'] = row[kind]['success'] - row['none']['success']
                    row[f'd_rmse_{kind}'] = row[kind]['rmse'] - row['none']['rmse']
                out.setdefault(suite, {})[nm] = row
    (common.RESULTS / 'reliance').mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(common.RESULTS / 'reliance' / f'{group}.json', out)
    return dict(suites=list(out))


# ------------------------------------------------------------------- probes
PCA_DIMS = 16


def _ridge_r2(X, Y, groups, rng, pca=None):
    """Ridge with alpha by 5-fold CV (episode folds) on an 80/20 episode split. pca: project the
    state on its first `pca` principal components (fit on the training split) first."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    test_eps = set(uniq[:max(1, len(uniq) // 5)])
    te = np.array([g in test_eps for g in groups])
    Xtr, Ytr, Xte, Yte = X[~te], Y[~te], X[te], Y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if pca is not None and Xtr.shape[1] > pca:
        _, _, Vt = np.linalg.svd(Xtr, full_matrices=False)
        P = Vt[:pca].T
        Xtr, Xte = Xtr @ P, Xte @ P
    ym = Ytr.mean(0)
    gtr = groups[~te]
    folds = np.array_split(np.unique(gtr), 5)

    def fit(Xa, Ya, a):
        return np.linalg.solve(Xa.T @ Xa + a * np.eye(Xa.shape[1]), Xa.T @ (Ya - ym))
    best, best_a = -np.inf, 1.0
    for a in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        sc = []
        for fo in folds:
            m = np.isin(gtr, fo)
            if m.all() or not m.any():
                continue
            W = fit(Xtr[~m], Ytr[~m], a)
            pred = Xtr[m] @ W + ym
            sc.append(1 - ((pred - Ytr[m]) ** 2).sum() / max(((Ytr[m] - Ytr[m].mean(0)) ** 2).sum(), 1e-12))
        if sc and np.mean(sc) > best:
            best, best_a = np.mean(sc), a
    W = fit(Xtr, Ytr, best_a)
    pred = Xte @ W + ym
    return float(1 - ((pred - Yte) ** 2).sum() / max(((Yte - Yte.mean(0)) ** 2).sum(), 1e-12)), best_a


def probes(ctx, group):
    sets = load_group(group)
    prof = ctx.profile
    n = prof['probe_episodes']
    out = {}
    for ms in sets:
        for suite in ('val_id', 'delay_100'):
            res = runner.evaluate(ms, suite, n, eval_seed=prof['eval_seed'] + 5, record=('probe',),
                                  chunk=prof['eval_chunk'], stride=10)
            pr = res['probe']
            n_out = pr['hidden'].shape[2]
            t_idx = np.arange(n_out)
            out.setdefault(suite, {})
            for m, name in enumerate(ms.names):
                mask = pr['alive'][m].astype(bool) & (t_idx[None] >= 10)
                if mask.sum() < 50:
                    out[suite][name] = None
                    continue
                ep = np.broadcast_to(np.arange(n)[:, None], mask.shape)[mask]
                X = pr['hidden'][m][mask]
                targets = dict(v_true=pr['v_true'][m][mask], v_delivered=pr['v_deliv'][m][mask],
                               t2w=pr['t2w'][m][mask][:, None], f_ext=pr['f_ext'][m][mask],
                               act_delay=pr['act_delay'][m][mask][:, None])
                row = {}
                for k, Y in targets.items():
                    row[k] = _ridge_r2(X, Y, ep, np.random.default_rng(0))[0]
                    row[f'{k}_pca{PCA_DIMS}'] = _ridge_r2(X, Y, ep, np.random.default_rng(0), pca=PCA_DIMS)[0]
                row['state_dim'] = int(X.shape[1])
                out[suite][name] = row
    (common.RESULTS / 'probes').mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(common.RESULTS / 'probes' / f'{group}.json', out)
    return dict(suites=list(out))
