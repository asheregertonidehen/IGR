"""Build eval ModelSets for E, T, and trained student families.

Students are always loaded from their SELECTED (best-val_id) checkpoints, written by the
select_ckpt_<family> job; E and T from their gate-selected checkpoints. A family with
several static estimator modes (F2a, F2b) is returned as one ModelSet per mode.
"""
import json

import jax
import jax.numpy as jnp

from eval.runner import ModelSet
from models import variants as V
from models.expert import Expert
from train import common

EXPERT_CLEAN = Expert(corrupt=False)
EXPERT_CORRUPT = Expert(corrupt=True)
_ARCH = {}


def _expand(tree):
    return jax.tree.map(lambda x: x[None], tree)


def lambda_star():
    p = common.RESULTS / 'selection.json'
    return json.loads(p.read_text())['lambda_star'] if p.exists() else None


def est_tuning():
    p = common.RESULTS / 'tune_classical' / 'selection.json'
    return json.loads(p.read_text()) if p.exists() else None


def expert_set(params, corrupt=False, name='E'):
    """One expert (unstacked params)."""
    return ModelSet(names=[name], arch=EXPERT_CORRUPT if corrupt else EXPERT_CLEAN, params=_expand(params),
                    cfg={'input_kind': jnp.zeros(1)})


def expert_stack(params, names, corrupt=False):
    return ModelSet(names=list(names), arch=EXPERT_CORRUPT if corrupt else EXPERT_CLEAN, params=params,
                    cfg={'input_kind': jnp.zeros(len(names))})


def teacher_arch(program='T', cell='gru'):
    k = ('T', program, cell)
    if k not in _ARCH:
        _ARCH[k] = V.teacher_arch(program, cell)
    return _ARCH[k]


def teacher_set(params, names, program='T'):
    cfgs = [V.teacher_cfg(n) for n in names]
    cfg = {k: jnp.stack([c[k] for c in cfgs]) for k in cfgs[0]}
    return ModelSet(names=list(names), arch=teacher_arch(program), params=params, cfg=cfg)


def fam_arch(fam):
    if fam not in _ARCH:
        _ARCH[fam] = V.family_arch(fam)
    return _ARCH[fam]


def family_set(fam, stacked_params, models, lam=None, tuning=None):
    """models: [(name, variant, ...)] aligned with the leading params axis. All models must share
    one estimator mode (use `by_est` to split)."""
    variants = [m[1] for m in models]
    modes = {V.VARIANT_EST.get(v) for v in variants}
    assert len(modes) == 1, f'{fam}: mixed estimator modes {modes}; split with by_est'
    lam = lambda_star() if lam is None else lam
    tuning = est_tuning() if tuning is None else tuning
    return ModelSet(names=[m[0] for m in models], arch=fam_arch(fam), params=stacked_params,
                    cfg=V.stack_cfg(variants, deploy=True, lam_star=lam, est_tuning=tuning),
                    meta=dict(family=fam, models=[list(m) for m in models]), est=modes.pop())


def by_est(models):
    """{est_mode: [indices]} for a list of model tuples."""
    groups = {}
    for i, m in enumerate(models):
        groups.setdefault(V.VARIANT_EST.get(m[1]), []).append(i)
    return groups


def family_sets(fam, stacked_params, models, lam=None, tuning=None):
    out = []
    for _, idx in by_est(models).items():
        sub = jax.tree.map(lambda x: x[jnp.asarray(idx)], stacked_params)
        out.append(family_set(fam, sub, [models[i] for i in idx], lam, tuning))
    return out


def selected_path(fam):
    return common.RESULTS / f'select_ckpt_{fam}' / 'params.pkl'


def load_family(fam):
    params, meta = common.load_params(selected_path(fam))
    return family_sets(fam, params, [tuple(m) for m in meta['models']])


def load_variant(variant):
    """Every student of a variant from its families' selected checkpoints, as a list of ModelSets.
    'Full' = the lambda_s* model of F1_S1 (student seed 0, renamed) + Full of F1_S2."""
    lam = lambda_star()
    out = []
    for fam in V.families_of_variant(variant):
        p = selected_path(fam)
        if not p.exists():
            continue
        params, meta = common.load_params(p)
        models = [tuple(m) for m in meta['models']]
        want = V.lam_name(lam) if (variant == 'Full' and fam == 'F1_S1' and lam is not None) else variant
        idx = [i for i, m in enumerate(models) if m[1] == want]
        if not idx:
            continue
        sub = jax.tree.map(lambda x: x[jnp.asarray(idx)], params)
        mods = [(V.model_name('Full', m[2], m[3]) if want != variant else m[0], m[1]) + tuple(m[2:])
                for m in (models[i] for i in idx)]
        out += family_sets(fam, sub, mods)
    return out


def load_teachers(program):
    params, meta = common.load_params(common.RESULTS / program / 'params.pkl')
    return teacher_set(params, [m[0] for m in meta['models']], program)


def load_experts():
    params, meta = common.load_params(common.RESULTS / 'E' / 'params.pkl')
    return expert_stack(params, [m[0] for m in meta['models']])


def subset(ms, idx, names=None):
    return ModelSet(names=names or [ms.names[i] for i in idx], arch=ms.arch,
                    params=jax.tree.map(lambda x: x[jnp.asarray(idx)], ms.params),
                    cfg={k: v[jnp.asarray(idx)] for k, v in ms.cfg.items()}, out=ms.out, meta=ms.meta, est=ms.est)
