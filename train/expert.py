"""Stage E: parameter-conditioned privileged experts E_0..E_2 in one vmapped PPO program.

Clean full-state inputs (sensing corruption is invisible to E) but trained WITH dynamics events
and actuation delay so its labels are correct for laggy or weakened vehicles. Difficulty-weighted
drone resampling. Quality gate per seed on `val` drones, clean, hold + slow Langevin:
success >= 95 % and Langevin RMSE < 0.15 m; seeds that miss it continue in +10 % chunks up to
+50 % (the whole program continues; each seed keeps its best checkpoint by gate score, a passing
seed keeps its first passing checkpoint).

Unsolved-drone mask: each E_k flies every training drone (training distribution, 5 s episodes,
`unsolved_reps` episodes per drone); drones with success < 80 % are masked from every imitation
label of teacher seed k (E_k and T_k). More than 10 % masked flags the run as degraded.
"""
import jax
import jax.numpy as jnp
import numpy as np

from eval import modelsets, runner
from models.expert import Expert
from sim import specs
from train import common, ppo

HP = dict(lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4, vf_coef=0.5)
UNSOLVED_SUCCESS = 0.8
DEGRADED_FRAC = 0.1


def gate(params_1, n_eps, seed):
    res = runner.evaluate(modelsets.expert_set(params_1), 'gate_E', n_eps, eval_seed=seed)
    df = runner.metrics_table(res)
    lang = df[df['task'] == 1]
    out = dict(success=float(df['success'].mean()),
               rmse_langevin=float(np.nanmean(lang['rmse'])) if lang['rmse'].notna().any() else float('inf'),
               rmse_all=float(np.nanmean(df['rmse'])) if df['rmse'].notna().any() else float('inf'))
    out['passed'] = bool(out['success'] >= 0.95 and out['rmse_langevin'] < 0.15)
    out['score'] = out['success'] - 0.5 * min(out['rmse_langevin'], 1.0)
    return out


def unsolved(params_1, reps, seed, chunk):
    """Per-train-drone success of one expert on the training distribution (5 s episodes)."""
    spec, _ = specs.train_spec()
    dset = common.drone_set('train')
    n_d = int(dset['mass'].shape[0])
    idx = jnp.tile(jnp.arange(n_d), reps)
    pool = common.sample_pool(jax.random.PRNGKey(seed), dset, spec, 1.0, n_d * reps, drone_idx=idx)
    res = runner.evaluate(modelsets.expert_set(params_1), 'unsolved_train', n_d * reps, spec=spec,
                          meta={'drones': 'train'}, pool=pool, steps=500, eval_seed=seed, chunk=min(chunk, n_d * reps))
    succ = res['rec']['alive'][0, :, -1].astype(float).reshape(reps, n_d).mean(0)
    return succ < UNSOLVED_SUCCESS, succ


def run(ctx):
    b, prof = ctx.budget, ctx.profile
    seeds = list(prof['teacher_seeds'])
    M = len(seeds)
    names = [f'E_{k}' for k in seeds]
    spec, _ = specs.train_spec()
    trainer = ppo.PPO(Expert(), {}, common.drone_set('train'), spec, b['n_envs'], b.get('rollout', 32), HP, M,
                      resample=True)
    key = jax.random.PRNGKey(10)
    params = trainer.init_params(jnp.stack([jax.random.PRNGKey(1000 + k) for k in seeds]))
    total = int(b['steps'])
    sel_seed = prof['selection_seed']

    def schedule(frac):
        return dict(sev=common.severity(frac), p_truth=0.0, alpha=0.0, lr=common.lr_linear(HP['lr'], frac) + 1e-5)

    params, opt, step, info = ppo.train_loop(ctx, trainer, params, total, schedule, key, names, tag='E',
                                             stage_steps=total)
    n_gate = prof['gate_episodes']
    take = lambda p, i: jax.tree.map(lambda x: x[i], p['actor'])
    best = [None] * M
    history = {n: [] for n in names}

    def check(params, step):
        for i, n in enumerate(names):
            g = gate(take(params, i), n_gate, sel_seed)
            history[n].append(dict(step=step, **g))
            ctx.log(f'gate {n} @ {step}: {g}')
            if best[i] is None or (not best[i][0]['passed'] and (g['passed'] or g['score'] > best[i][0]['score'])):
                best[i] = (g, take(params, i), step)
    check(params, step)
    extra = 0
    while not all(bst[0]['passed'] for bst in best) and extra < 5:
        extra += 1
        target = step + int(0.1 * total)
        sched2 = lambda frac: dict(sev=1.0, p_truth=0.0, alpha=0.0, lr=0.3 * HP['lr'])
        params, opt, step, info = ppo.train_loop(ctx, trainer, params, target, sched2, jax.random.fold_in(key, extra),
                                                 names, tag=f'E_cont{extra}', stage_steps=target, start_step=step,
                                                 opt_state=opt, failed=info['failed_mask'])
        check(params, step)
    stacked = jax.tree.map(lambda *x: jnp.stack(x), *[bst[1] for bst in best])
    masks, succ = [], []
    for i, n in enumerate(names):
        m, s = unsolved(best[i][1], int(prof.get('unsolved_reps', 4)), sel_seed + 31, 4096)
        masks.append(m)
        succ.append(s)
    masks = np.stack(masks)
    np.save(ctx.job_dir / 'unsolved.npy', masks)
    np.save(ctx.job_dir / 'train_drone_success.npy', np.stack(succ))
    frac = masks.mean(1)
    meta = dict(models=[[n, 'E', k, 0] for n, k in zip(names, seeds)], seeds=seeds,
                gate={n: best[i][0] for i, n in enumerate(names)}, steps={n: best[i][2] for i, n in enumerate(names)},
                history=history, flagged={n: not best[i][0]['passed'] for i, n in enumerate(names)},
                unsolved_frac={n: float(f) for n, f in zip(names, frac)},
                degraded=bool((frac > DEGRADED_FRAC).any()), nan_counts=info['nan_counts'], failed=info['failed'])
    for n in names:
        if meta['flagged'][n]:
            ctx.note(f'{n} quality gate NOT met; proceeding with its best checkpoint (flagged in RESULTS.md)')
    if meta['degraded']:
        ctx.note(f'unsolved-drone fraction > {DEGRADED_FRAC:.0%} for some E seed: run flagged as DEGRADED ({meta["unsolved_frac"]})')
    common.save_params(ctx.job_dir / 'params.pkl', stacked, meta)
    common.atomic_write_json(ctx.job_dir / 'meta.json', meta)
    return dict(gate=meta['gate'], unsolved_frac=meta['unsolved_frac'], degraded=meta['degraded'])
