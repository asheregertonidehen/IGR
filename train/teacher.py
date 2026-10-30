"""Stage T: constrained teachers T_0..T_2 as one vmapped PPO program (3x wider cell).

The descriptor and the annealed truth channel reach GRU_ctl only.

Recurrent PPO (64-step truncated BPTT, stored initial carries, lr 2e-4), asymmetric critic,
imitation of the model's E_k with alpha(frac) = max(0.05, exp(-k frac)) (0.05 at 50 %) masked on
E_k's unsolved drones, event-weighted representation losses (pred 1, truth 1 incl. the P_c NLL,
id 0.5; no lambda sweep), truth annealed p_truth 1 -> 0 over the first 60 % (per episode).

Checkpoint choice: selection checkpoints every 10 % of the budget; each is evaluated on val_id
(selection seed) and the best by success (ties: lower RMSE) is kept per model. The quality gate
is evaluated on it: val_id success >= 90 % and >= E_k through corrupted sensing, clean
>= 90 %. Fallback chain per failing model:
  (1) continue with alpha frozen at 0.3 for +30 % budget;
  (2) DAgger-style imitation of E under constraints with truth annealing (same losses, no policy
      gradient) for 50 % budget, then PPO for the remaining 50 %.
The first passing candidate (else the best gate score) is kept; the path is recorded per model.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from eval import modelsets, runner
from models import inputs as I
from models import variants as V
from models.expert import Expert
from sim import specs
from train import common, ppo

HP = dict(lr=2e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4, vf_coef=0.5, pg_coef=1.0)
REP = dict(pred=1.0, truth=1.0, pc=1.0, id=0.5, beta=4.0)
SEL_FRAC = 0.1


def alpha_schedule(frac):
    return max(0.05, math.exp(math.log(0.05) * frac / 0.5))


def truth_schedule(frac):
    return max(0.0, 1.0 - frac / 0.6)


def select_eval(ms, n_eps, seed):
    """Per-model (success, rmse) on val_id with the selection seed."""
    df = runner.metrics_table(runner.evaluate(ms, 'val_id', n_eps, eval_seed=seed))
    g = df.groupby('model')
    return {m: (float(g.get_group(m).success.mean()), float(np.nanmean(g.get_group(m).rmse))
                if g.get_group(m).rmse.notna().any() else float('inf')) for m in ms.names}


def gate(t_ms, e_params, n_eps, seed):
    """Gate per teacher model (t_ms: teacher ModelSet; e_params: stacked E params aligned with it)."""
    out = {n: {} for n in t_ms.names}
    for suite in ('val_id', 'clean'):
        df = runner.metrics_table(runner.evaluate(t_ms, suite, n_eps, eval_seed=seed))
        for n, g in df.groupby('model'):
            out[n][f'T_{suite}'] = float(g.success.mean())
    e_ms = modelsets.expert_stack(e_params, t_ms.names, corrupt=True)
    df = runner.metrics_table(runner.evaluate(e_ms, 'val_id', n_eps, eval_seed=seed))
    for n, g in df.groupby('model'):
        out[n]['E_corrupt_val_id'] = float(g.success.mean())
    for o in out.values():
        o['passed'] = bool(o['T_val_id'] >= 0.9 and o['T_val_id'] >= o['E_corrupt_val_id'] and o['T_clean'] >= 0.9)
        o['score'] = o['T_val_id'] + 0.5 * o['T_clean']
    return out


def _take(tree, idx):
    return jax.tree.map(lambda x: x[jnp.asarray(idx)], tree)


def run(ctx, program='T'):
    b, prof = ctx.budget, ctx.profile
    names = list(V.TEACHER_PROGRAMS[program])
    if program == 'T' and len(prof['teacher_seeds']) < 3:      # smoke profiles may use fewer seeds
        names = [n for n in names if not n[2:].isdigit() or int(n[2:]) in prof['teacher_seeds']]
    M = len(names)
    tinfo = [V.TEACHERS[n] for n in names]
    spec, _ = specs.train_spec()
    I.target_stats(required=True)
    arch = modelsets.teacher_arch(program)
    e_all, e_meta = common.load_params(common.RESULTS / 'E' / 'params.pkl')
    e_seeds = [m[2] for m in e_meta['models']]
    e_idx = [e_seeds.index(t['eseed']) for t in tinfo]
    lab_params = _take(e_all, e_idx)
    um = common.unsolved_mask(prof)
    n_dr = int(common.drone_set('train')['mass'].shape[0])
    lab_mask = np.stack([~um[i] for i in e_idx]) if um is not None else np.ones((M, n_dr), bool)
    cfg_rows = [V.teacher_cfg(n) for n in names]
    cfg = {k: jnp.stack([c[k] for c in cfg_rows]) for k in cfg_rows[0]}
    total = int(b['steps'])
    sel_store = common.SelectionStore(ctx.job_dir)
    rep = {k: jnp.full(M, v) for k, v in REP.items()}

    def make(idx, hp):
        sub = lambda d: {k: v[jnp.asarray(idx)] for k, v in d.items()}
        return ppo.PPO(arch, sub(cfg), common.drone_set('train'), spec, b['n_envs'], b.get('chunk', 64), hp, len(idx),
                       labeler=(Expert(), {}), rep=sub(rep))

    trainer = make(list(range(M)), HP)
    key = jax.random.PRNGKey(20)
    params = trainer.init_params(jnp.stack([jax.random.PRNGKey(2000 + t['init']) for t in tinfo]))

    def sched_main(frac):
        return dict(sev=common.severity(frac), p_truth=np.full(M, truth_schedule(frac)),
                    alpha=alpha_schedule(frac), lr=common.lr_linear(HP['lr'], frac) + 1e-5)

    params, opt, step, info = ppo.train_loop(ctx, trainer, params, total, sched_main, key, names, lab_params=lab_params,
                                             lab_mask=lab_mask, tag='T_main', stage_steps=total,
                                             sel_every=max(1, int(SEL_FRAC * total)), sel_store=sel_store)
    # ---- best-val_id selection checkpoint per model
    n_sel, seed = prof['selection_episodes'], prof['selection_seed']
    hist = {n: [] for n in names}
    best = {}
    for s in sel_store.steps('T_main'):
        p_s = sel_store.load('T_main', s)
        res = select_eval(modelsets.teacher_set(p_s, names, program), n_sel, seed)
        for i, n in enumerate(names):
            hist[n].append(dict(step=s, success=res[n][0], rmse=res[n][1]))
            if n not in best or (res[n][0], -res[n][1]) > (best[n][0], -best[n][1]):
                best[n] = (res[n][0], res[n][1], s, jax.tree.map(lambda x: x[i], p_s))
        ctx.log(f'T selection @ {s}: ' + ', '.join(f'{n} {res[n][0]:.3f}' for n in names))
    chosen = jax.tree.map(lambda *x: jnp.stack(x), *[best[n][3] for n in names])
    n_gate = prof['gate_episodes']
    g = gate(modelsets.teacher_set(chosen, names, program), lab_params, n_gate, seed + 1)
    cands = {n: [('main', g[n], best[n][3], best[n][2])] for n in names}
    for n in names:
        ctx.log(f'gate {n} (main, step {best[n][2]}): {g[n]}')
    # ---- fallback chain for failing models
    fail = [i for i, n in enumerate(names) if not g[n]['passed']]
    if fail:
        fn = [names[i] for i in fail]
        ctx.note(f'T gate failed for {fn}: fallback 1 (alpha = 0.3, +30 %)')
        tr1 = make(fail, HP)
        s1 = lambda frac: dict(sev=1.0, p_truth=0.0, alpha=0.3, lr=0.5 * HP['lr'])
        p_in = dict(actor=_take(chosen, fail), critic=_take(params['critic'], fail))
        p1, _, st1, _ = ppo.train_loop(ctx, tr1, p_in, int(0.3 * total), s1, jax.random.fold_in(key, 2), fn,
                                       lab_params=_take(lab_params, fail), lab_mask=lab_mask[fail], tag='T_fb1')
        g1 = gate(modelsets.teacher_set(p1['actor'], fn, program), _take(lab_params, fail), n_gate, seed + 1)
        for j, n in enumerate(fn):
            cands[n].append(('fallback1_alpha0.3', g1[n], jax.tree.map(lambda x: x[j], p1['actor']), step + st1))
            ctx.log(f'gate {n} (fallback 1): {g1[n]}')
        fail2 = [i for i, n in zip(fail, fn) if not g1[n]['passed']]
        if fail2:
            fn2 = [names[i] for i in fail2]
            ctx.note(f'T gate failed for {fn2}: fallback 2 (DAgger 50 % + PPO 50 %)')
            dag = make(fail2, dict(HP, pg_coef=0.0, lr=5e-4))
            half = int(0.5 * total)
            pd = dag.init_params(jnp.stack([jax.random.PRNGKey(3000 + tinfo[i]['init']) for i in fail2]))
            sd = lambda frac: dict(sev=common.severity(frac), p_truth=truth_schedule(frac), alpha=1.0,
                                   lr=common.lr_linear(5e-4, frac) + 1e-5)
            pd, _, _, _ = ppo.train_loop(ctx, dag, pd, half, sd, jax.random.fold_in(key, 4), fn2,
                                         lab_params=_take(lab_params, fail2), lab_mask=lab_mask[fail2], tag='T_fb2_dagger')
            tr2 = make(fail2, HP)
            sf = lambda frac: dict(sev=1.0, p_truth=0.0, alpha=0.1, lr=common.lr_linear(HP['lr'], frac) + 1e-5)
            pd, _, _, _ = ppo.train_loop(ctx, tr2, pd, total - half, sf, jax.random.fold_in(key, 5), fn2,
                                         lab_params=_take(lab_params, fail2), lab_mask=lab_mask[fail2], tag='T_fb2_ppo')
            g2 = gate(modelsets.teacher_set(pd['actor'], fn2, program), _take(lab_params, fail2), n_gate, seed + 1)
            for j, n in enumerate(fn2):
                cands[n].append(('fallback2_dagger_ppo', g2[n], jax.tree.map(lambda x: x[j], pd['actor']), total))
                ctx.log(f'gate {n} (fallback 2): {g2[n]}')
    final, meta_models = [], {}
    for n in names:
        passing = [c for c in cands[n] if c[1]['passed']]
        path, gg, p, st = passing[0] if passing else max(cands[n], key=lambda c: c[1]['score'])
        final.append(p)
        meta_models[n] = dict(path=path, gate=gg, step=int(st), flagged=not gg['passed'],
                              candidates=[dict(path=c[0], step=int(c[3]), **c[1]) for c in cands[n]],
                              selection=hist[n], selected_step=int(best[n][2]))
        if not gg['passed']:
            ctx.note(f'{n} quality gate NOT met by any path; using best ({path}), flagged in RESULTS.md')
    stacked = jax.tree.map(lambda *x: jnp.stack(x), *final)
    meta = dict(models=[[n, 'T', V.TEACHERS[n]['eseed'], 0] for n in names], program=program, teachers=meta_models,
                deploy_params=arch.deploy_param_count(final[0], V.VARIANTS['Full']), nan_counts=info['nan_counts'],
                failed=info['failed'], steps=step)
    common.save_params(ctx.job_dir / 'params.pkl', stacked, meta)
    common.atomic_write_json(ctx.job_dir / 'meta.json', meta)
    return {n: dict(path=m['path'], **{k: m['gate'][k] for k in ('T_val_id', 'T_clean', 'E_corrupt_val_id', 'passed')})
            for n, m in meta_models.items()}
