"""Fixed per-dimension statistics for the representation-head targets.

Runs once after stage E: the deterministic expert E_0 flies the TRAINING distribution
(full severity, train drones) and the raw truth (25-d), effective identity, IMU-prediction
and P_c (true current external-channel value) targets are recorded; their mean and std are written to RESULTS/target_stats.npz and used by
models/inputs.py for every head (teacher and students alike, so capability
distillation compares like with like). E is used because it precedes every model
that has heads, so the statistics do not depend on any model they normalize.
"""
import jax
import jax.numpy as jnp
import numpy as np

from models import inputs as I
from models.expert import Expert
from sim import env as envm
from sim import specs
from train import common

EXPERT = Expert()


def run(ctx):
    b = ctx.profile.get('target_stats', dict(n_envs=1024, steps=500))
    spec, _ = specs.train_spec()
    dset = common.drone_set('train')
    e_all, _ = common.load_params(common.RESULTS / 'E' / 'params.pkl')
    e_params = jax.tree.map(lambda x: x[0], e_all)     # E_0
    n, T = int(b['n_envs']), int(b['steps'])
    key = jax.random.PRNGKey(7)
    env = jax.vmap(envm.reset)(common.sample_pool(key, dset, spec, 1.0, n))

    @jax.jit
    def roll(env, key):
        def one(env, k):
            pv = jax.vmap(envm.privileged)(env)
            a = jnp.tanh(EXPERT.mean(e_params, jax.vmap(I.expert)(env['obs'], pv)))
            rec = dict(truth=jax.vmap(I.truth_raw)(pv), id=jax.vmap(I.id_raw)(pv),
                       pred=jax.vmap(I.pred_raw)(env['obs']), pc=jax.vmap(I.pc_raw)(pv), alive=env['alive'])
            env2, _, _ = jax.vmap(envm.step)(env, a, jax.random.split(k, n))
            return env2, rec
        return jax.lax.scan(one, env, jax.random.split(key, T))[1]

    rec = jax.tree.map(np.asarray, roll(env, jax.random.fold_in(key, 1)))
    ok = rec.pop('alive').astype(bool)
    assert set(rec) == set(I.STAT_KEYS), sorted(rec)
    out = {}
    for k, x in rec.items():
        x = x[ok]
        out[f'{k}_mean'], out[f'{k}_std'] = x.mean(0), x.std(0)
    path = I._stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    I._TSTATS.clear()
    ctx.log('target stats: ' + '; '.join(f'{k} std {np.round(out[k + "_std"], 3).tolist()}' for k in rec))
    return {k: v.tolist() for k, v in out.items()}
