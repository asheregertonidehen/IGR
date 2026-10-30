"""The experiment as data: variants, teachers, families and seeds.

Paper names: `Full` = IGR-sup, `A3c` = IGR-task.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp

from models import baselines, gated_cell
from models import inputs as I

LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
LAM_DEFAULT = dict(lam_pred=1.0, lam_truth=1.0, lam_pc=1.0, lam_id=0.5, lam_T=0.5, lam_g=0.1)
_ZERO_LAM = {k: 0.0 for k in LAM_DEFAULT}
_BASE = dict(use_gates=1.0, use_innov=1.0, rep_grad=1.0, label=1.0, input_kind=float(I.KINDS['native']), sev_mult=1.0, lam_scale=1.0, beta=4.0,
             ekf_q=1.0, ekf_dhat=0.0, leak_tau=0.05, corr_scale=1.0, **LAM_DEFAULT)
CFG_KEYS = tuple(_BASE)
K = I.KINDS


def _v(**kw):
    d = dict(_BASE)
    d.update(kw)
    return d


def lam_name(lam):
    return f'Full_l{lam:g}'


VARIANTS = {
    # ---- GRU baselines
    'B4': _v(label=0.0, **_ZERO_LAM),                               # expert labels
    'B5m': _v(label=1.0, input_kind=float(K['ekf_matched']), **_ZERO_LAM),  # GRU fed by the Kalman filter
    'A7': _v(label=1.0, **_ZERO_LAM),                               # teacher labels
    'A9': _v(lam_pc=0.0, lam_g=0.0),                                # GRU with physical heads
    # ---- innovation-gated cell
    'Full': _v(),                                                   # IGR-sup
    'A3c': _v(lam_pc=0.0),                                          # IGR-task
    'A1': _v(label=0.0, lam_T=0.0, lam_g=0.0),                      # expert labels
    'A2': _v(use_gates=0.0),                                        # no gates
    'A3': _v(**_ZERO_LAM),                                          # no auxiliary objectives
    'A3a': _v(lam_pred=0.0),                                        # no inertial forecast
    'A3b': _v(lam_truth=0.0),                                       # no state or disturbance target
    'A4': _v(rep_grad=0.0),                                         # no control gradient into the representation
    'A5': _v(use_innov=0.0),                                        # raw embeddings, no innovation
    'A11': _v(lam_T=0.0, lam_g=0.0),                                # no capability or gate distillation
}
for _lam in LAMBDA_GRID:
    VARIANTS[lam_name(_lam)] = _v(lam_scale=_lam)
# Students use correction scale 0.5; teachers keep 1.0. lambda_s is fixed at the selected 0.01.
CORR_SCALE = 0.5
LAMBDA_S = 0.01
GATED = ('A1', 'A2', 'A3', 'A3a', 'A3b', 'A3c', 'A4', 'A5', 'A11')
for _n, _d in VARIANTS.items():
    if _n.startswith('Full') or _n in GATED:
        _d['corr_scale'] = CORR_SCALE
LAMBDA_SWEEP = [lam_name(LAMBDA_S)]
# variants whose lambda scale is the val_id-selected lambda_s* (stage S2); A9 too.
LAMBDA_STAR = set(GATED) | {'Full', 'A9'}
CORE = ('Full', 'A1', 'A3', 'A7', 'A9', 'B4', 'B5m')
# static classical-estimator mode per variant (sensing/layer.EST_MODES)
VARIANT_EST = {'B5m': 'nots'}
# tuned estimator parameter per variant (from the tune_classical job, selected on val_id)
EST_TUNED = {'B5m': 'ekf_nots'}

# ---- teachers: one vmapped wide program
TEACHER_WIDTH = 3
TEACHERS = {
    'T_0': dict(eseed=0, width=TEACHER_WIDTH, init=0),
    'T_1': dict(eseed=1, width=TEACHER_WIDTH, init=1),
    'T_2': dict(eseed=2, width=TEACHER_WIDTH, init=2),
}
TEACHER_PROGRAMS = {'T': ['T_0', 'T_1', 'T_2']}


def teacher_program_of(tname):
    return next(p for p, ts in TEACHER_PROGRAMS.items() if tname in ts)


# ---- student families; stage A runs before lambda selection, stage B after it
FAMILIES = {
    'F1_S1': dict(arch='gated', cell='gru', stage='A', teachers='T', variants=LAMBDA_SWEEP),
    'F2b': dict(arch='gru_matched', stage='A', teachers='T', variants=['B4', 'A7', 'B5m']),
    'F1_S2': dict(arch='gated', cell='gru', stage='B', teachers='T',
                  variants=['Full', 'A1', 'A2', 'A3', 'A3b', 'A3c', 'A4', 'A5', 'A3a', 'A11']),
    'F2b_A9': dict(arch='gru_matched', heads=True, stage='B', teachers='T', variants=['A9']),
}
STAGES = {s: [f for f, d in FAMILIES.items() if d['stage'] == s] for s in ('A', 'B')}


def model_name(variant, tseed, sseed):
    return f'{variant}_t{tseed}_s{sseed}'


def parse_model(name):
    """'Full_l0.3_t1_s0' -> ('Full_l0.3', 1, 0); names without seeds -> (name, 0, 0)."""
    parts = name.rsplit('_', 2)
    if len(parts) == 3 and parts[1][:1] == 't' and parts[2][:1] == 's' and parts[1][1:].isdigit() \
            and parts[2][1:].isdigit():
        return parts[0], int(parts[1][1:]), int(parts[2][1:])
    return name, 0, 0


def variant_seeds(variant, seeds):
    """[(teacher_seed, student_seed)] for a variant. `seeds` from configs/run.yaml."""
    ts = seeds['teacher_seeds']
    if variant in CORE:
        ss = seeds['student_seeds_core']
        if variant == 'Full':          # Full's student seed 0 is the lambda_s* model of stage S1
            ss = [s for s in ss if s != ss[0]]
        return [(k, s) for k in ts for s in ss]
    return [(k, s) for k in ts for s in seeds['student_seeds_other']]


def family_models(fam, seeds):
    """[(model_name, variant, teacher_seed, student_seed, teacher_name)] in vmapped order."""
    out = []
    for v in FAMILIES[fam]['variants']:
        pairs = ([(k, seeds['student_seeds_core'][0]) for k in seeds['teacher_seeds']] if v in LAMBDA_SWEEP
                 else variant_seeds(v, seeds))
        for k, s in pairs:
            out.append((model_name(v, k, s), v, k, s, f'T_{k}'))
    return out


def stack_cfg(variants, lam_star=None, est_tuning=None):
    """[M]-stacked config; lam_star sets the lambda scale of the gated students, est_tuning the filter settings."""
    rows = []
    for v in variants:
        d = dict(VARIANTS[v])
        if lam_star is not None and v in LAMBDA_STAR:
            d['lam_scale'] = float(lam_star)
        if est_tuning and v in EST_TUNED and EST_TUNED[v] in est_tuning:
            t = est_tuning[EST_TUNED[v]]
            d.update(ekf_q=t.get('q', d['ekf_q']), ekf_dhat=t.get('dhat', d['ekf_dhat']),
                     leak_tau=t.get('tau', d['leak_tau']))
        rows.append(d)
    return {k: jnp.asarray([r[k] for r in rows], jnp.float32) for k in CFG_KEYS}


def single_cfg(variant='Full', **over):
    d = dict(VARIANTS[variant], **over)
    return {k: jnp.asarray(d[k], jnp.float32) for k in CFG_KEYS}


def teacher_cfg(tname):
    return single_cfg('Full', corr_scale=1.0)


def est_params(cfg_m):
    """Per-model estimator tuning in the env (sim/env.with_est) from a (possibly [M]) config."""
    return dict(q=cfg_m['ekf_q'], dhat=cfg_m['ekf_dhat'], tau=cfg_m['leak_tau'])


@lru_cache(maxsize=None)
def full_deploy_params():
    cell = gated_cell.GatedCell('gru', teacher=False)
    return cell.deploy_param_count(cell.init(jax.random.PRNGKey(0)), VARIANTS['Full'])


@lru_cache(maxsize=None)
def matched_hidden():
    return baselines.matched_hidden(I.NATIVE_DIM, full_deploy_params())


def family_arch(fam):
    f = FAMILIES[fam]
    if f['arch'] == 'gated':
        return gated_cell.GatedCell(f['cell'], teacher=False)
    if f['arch'] == 'gru16':
        return baselines.GRUBaseline(16)
    return baselines.GRUBaseline(matched_hidden(), heads=f.get('heads', False))


def teacher_arch(program='T', cell='gru'):
    width = TEACHERS[TEACHER_PROGRAMS[program][0]]['width']
    return gated_cell.GatedCell(cell, teacher=True, width=width)


def deploy_params(fam, variant, params=None):
    arch = family_arch(fam)
    p = params if params is not None else arch.init(jax.random.PRNGKey(0))
    return arch.deploy_param_count(p, VARIANTS[variant])


def family_of(variant):
    for fam, f in FAMILIES.items():
        if variant in f['variants']:
            return fam
    raise KeyError(variant)


def families_of_variant(variant):
    """Families holding students of a variant (Full: lambda_s* model of F1_S1 + F1_S2)."""
    if variant == 'Full':
        return ['F1_S1', 'F1_S2']
    return [family_of(variant)]
