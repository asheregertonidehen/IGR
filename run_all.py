"""Run the full pipeline.

    python run_all.py --profile smoke    # whole pipeline at tiny sizes on CPU
    python run_all.py --profile full     # the reported experiment (resumable; GPU)

Outputs: full -> results/, logs/; smoke -> results/_smoke/, logs/_smoke/. Jobs are idempotent
(results/<job>/done.json), resume from checkpoints, retry once on an exception and then stop their
dependents. Stages run to their fixed budgets.
"""
import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', choices=('smoke', 'full'), default='full')
    ap.add_argument('--only', nargs='*', help='run only these jobs')
    return ap.parse_args()


def setup_env(profile):
    sub = '' if profile == 'full' else f'/_{profile}'
    os.environ.setdefault('IGR_RESULTS', str(ROOT / f'results{sub}'))
    os.environ.setdefault('IGR_LOGS', str(ROOT / f'logs{sub}'))
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.92')
    sys.path.insert(0, str(ROOT))


class Logger:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self.lock:
            print(line, flush=True)
            with open(self.path, 'a') as f:
                f.write(line + '\n')


def manifest(common, profile_name, prof):
    import jax
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout
    hw = dict(platform=platform.platform(), python=sys.version, jax=jax.__version__, devices=[str(d) for d in jax.devices()])
    configs = {p.name: p.read_text() for p in (ROOT / 'configs').glob('*.yaml')}
    common.atomic_write_json(common.RESULTS / 'manifest.json', dict(profile=profile_name, resolved_profile=prof, hardware=hw,
                                                                    pip_freeze=freeze.splitlines(), configs=configs, started=time.time()))


def resolve_profile(cfg, name):
    prof = copy.deepcopy(cfg['profiles'][name])
    prof.update(cfg['seeds'], eval_seed=cfg['eval_seed'], selection_seed=cfg['selection_seed'],
                convergence=cfg['convergence'], classical_sweeps=cfg['classical_sweeps'])
    return prof


def check_config(cfg):
    from models import variants as V
    assert tuple(cfg['lambda_grid']) == tuple(V.LAMBDA_GRID), 'run.yaml lambda_grid != models/variants.LAMBDA_GRID'
    assert cfg['teacher_width']['main'] == V.TEACHER_WIDTH


def main():
    args = parse()
    setup_env(args.profile)
    import yaml
    from eval import runner
    from orchestrator import dag
    from orchestrator.context import JobContext
    from train import common

    cfg = yaml.safe_load((ROOT / 'configs' / 'run.yaml').read_text())
    check_config(cfg)
    log = Logger(common.LOGS / 'run.log')
    log(f'=== run_all profile={args.profile} results={common.RESULTS} ===')
    prof = resolve_profile(cfg, args.profile)
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    manifest(common, args.profile, prof)
    runner.DEFAULT_STEPS = prof.get('eval_steps')

    state = {}
    for name, job in dag.build(prof).items():
        if args.only and name not in args.only:
            continue
        job_dir = common.RESULTS / name
        done_file = job_dir / 'done.json'
        if done_file.exists():
            state[name] = 'done'
            continue
        if any(state.get(d) == 'failed' or (d not in state and not (common.RESULTS / d / 'done.json').exists()) for d in job.deps):
            state[name] = 'failed'
            log(f'{name}: BLOCKED')
            continue
        budget = copy.deepcopy(prof.get(job.budget_key, {})) if job.budget_key else {}
        log(f'--- {name} start')
        t0, err = time.time(), None
        for attempt in (1, 2):
            ctx = JobContext(job=name, job_dir=job_dir, profile=prof, budget=budget, log_fn=log)
            try:
                result = job.fn(ctx)
                err = None
                break
            except Exception:
                err = traceback.format_exc()
                log(f'{name}: exception (attempt {attempt}):\n{err}')
        wall = time.time() - t0
        if err is None:
            state[name] = 'done'
            common.atomic_write_json(done_file, dict(wall_s=wall, budget=budget, notes=ctx.notes, result=result, finished=time.time()))
            log(f'--- {name} done in {wall / 60:.1f} min')
        else:
            state[name] = 'failed'
            common.atomic_write_json(job_dir / 'failed.json', dict(traceback=err, wall_s=wall, notes=ctx.notes))
            log(f'--- {name} FAILED after {wall / 60:.1f} min')
    failed = [k for k, v in state.items() if v == 'failed']
    log(f'=== run_all finished: {len(state) - len(failed)} done, {len(failed)} failed {failed} ===')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
