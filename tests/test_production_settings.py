"""The full run must use explicit student optimization settings, never library or code defaults."""
import jax
import jax.numpy as jnp
import pytest
import yaml

import run_all
from models import variants as V
from train import distill


RUN_CFG = yaml.safe_load((run_all.ROOT / 'configs' / 'run.yaml').read_text())


def _profile(name='full'):
    cfg = yaml.safe_load((run_all.ROOT / 'configs' / 'run.yaml').read_text())
    return run_all.resolve_profile(cfg, name)


def test_full_profile_uses_the_reported_horizons_and_constant_lr_1e4():
    b = _profile('full')['student']
    assert b['bptt'] == 100 and b['ep_steps'] == 500   # 100-step backward window, state carried
    assert b['lr'] == b['lr_final'] == 1e-4
    assert distill.check_settings(b) == dict(bptt=100, lr=1e-4, lr_final=1e-4)
    assert all(distill.lr_at(b, f) == pytest.approx(1e-4) for f in (0.0, 0.1, 0.3, 1.0, 1.5))   # incl. extensions
    assert distill.family_bptt(b, 'F2b') == 500                  # reused baselines keep their trained horizon
    assert {distill.family_bptt(b, f) for f in ('F1_S1', 'F1_S2', 'F2b_A9')} == {100}


def test_smoke_profile_exercises_truncated_bptt_too():
    b = _profile('smoke')['student']
    assert b['bptt'] and b['ep_steps'] % b['bptt'] == 0 and b['bptt'] < b['ep_steps']


def test_missing_settings_stop_the_job_instead_of_defaulting():
    for missing in ('lr', 'lr_final', 'bptt'):
        b = {k: v for k, v in _profile('full')['student'].items() if k != missing}
        with pytest.raises(RuntimeError, match=missing):
            distill.check_settings(b)


def test_bptt_must_divide_the_episode_length():
    tp = jax.vmap(V.teacher_arch('T').init)(jnp.stack([jax.random.PRNGKey(i) for i in range(4)]))
    from models.expert import Expert
    e = jax.vmap(Expert().init)(jnp.stack([jax.random.PRNGKey(i) for i in range(3)]))
    seeds = dict(teacher_seeds=[0], student_seeds_core=[0, 1], student_seeds_other=[0])
    models = [m for m in V.family_models('F1_S2', seeds) if m[1] == 'Full'][:1]
    with pytest.raises(ValueError, match='must divide'):
        distill.Distiller('F1_S2', models, 2, tp, list(V.TEACHER_PROGRAMS['T']), e, [0, 1, 2], ep_steps=12, mb_seqs=1,
                          bptt=5)
