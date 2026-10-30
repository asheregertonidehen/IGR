"""Job DAG, executed in insertion order.

Experts -> target statistics -> filter tuning -> teachers
-> stage A (first IGR-sup students, plain baselines) -> checkpoint selection -> lambda
-> stage B (remaining students, A9) -> checkpoint selection -> evaluation, paired events, reliance, probes.
"""
import subprocess
import sys
from dataclasses import dataclass, field

from models import variants as V
from train import common


@dataclass
class Job:
    name: str
    fn: object
    deps: list = field(default_factory=list)
    budget_key: str = None
    kind: str = 'train'        # train | eval | analysis | test


def _unit_tests(ctx):
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', 'tests'], cwd=str(common.ROOT),
                       capture_output=True, text=True)
    (ctx.job_dir / 'pytest.txt').write_text(r.stdout + r.stderr)
    tail = (r.stdout.strip().splitlines() or ['?'])[-1]
    if r.returncode != 0:
        raise RuntimeError(f'unit tests failed: {tail}')
    return dict(result=tail)


def build(profile):
    from eval import jobs as ej
    from train import distill, expert, target_stats, teacher

    J = []
    add = lambda *a, **k: J.append(Job(*a, **k))
    ev = lambda g: (lambda c: ej.eval_group(c, g))
    add('unit_tests', _unit_tests, kind='test')
    add('eval_B0', ev('B0'), kind='eval')
    add('E', expert.run, [], 'E')
    add('target_stats', target_stats.run, ['E'], kind='analysis')
    add('tune_classical', ej.tune_classical, ['E'], kind='analysis')
    add('T', lambda c: teacher.run(c, 'T'), ['E', 'target_stats'], 'T')
    add('events_T', lambda c: ej.discrimination(c, 'T'), ['T'], kind='analysis')
    for fam in V.STAGES['A']:
        add(fam, lambda c, f=fam: distill.run(c, f), ['T', 'tune_classical'], 'student')
        add(f'select_ckpt_{fam}', lambda c, f=fam: ej.select_family(c, f), [fam], 'student', kind='analysis')
    add('converge_A', lambda c: ej.converge(c, 'A'), [f'select_ckpt_{f}' for f in V.STAGES['A']], 'student')
    add('select_lambda', ej.select_lambda, ['converge_A'], kind='analysis')
    for fam in V.STAGES['B']:
        add(fam, lambda c, f=fam: distill.run(c, f), ['T', 'select_lambda', 'tune_classical'], 'student')
        add(f'select_ckpt_{fam}', lambda c, f=fam: ej.select_family(c, f), [fam], 'student', kind='analysis')
    add('converge_B', lambda c: ej.converge(c, 'B'), [f'select_ckpt_{f}' for f in V.STAGES['B']], 'student')
    stud_dep = lambda fam: [f'select_ckpt_{fam}', 'converge_B']
    for g, deps in (('E', ['E']), ('T', ['T'])):
        add(f'eval_{g}', ev(g), deps, kind='eval')
    for fam in ('F1_S1', 'F1_S2', 'F2b', 'F2b_A9'):
        add(f'eval_{fam}', ev(fam), stud_dep(fam), kind='eval')
    for fam in ('F1_S1', 'F1_S2', 'F2b'):
        add(f'events_{fam}', lambda c, f=fam: ej.discrimination(c, f), stud_dep(fam), kind='analysis')
    for g in ('F1_S1', 'F1_S2', 'F2b', 'T'):
        add(f'reliance_{g}', lambda c, g=g: ej.reliance(c, g), ['T'] if g == 'T' else stud_dep(g), kind='analysis')
    for g in ('B0', 'T', 'F1_S1', 'F1_S2', 'F2b', 'F2b_A9'):
        deps = [] if g == 'B0' else (['T'] if g == 'T' else stud_dep(g))
        add(f'probes_{g}', lambda c, g=g: ej.probes(c, g), deps, kind='analysis')
    names = [j.name for j in J]
    assert len(names) == len(set(names)), 'duplicate job names'
    return {j.name: j for j in J}
