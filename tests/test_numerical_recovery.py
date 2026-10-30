"""Numerical regressions independent of simulator size and training outcomes."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from train import common, distill


@pytest.mark.parametrize('magnitude', [1.0, 1e12, 1e30, np.finfo(np.float32).max])
def test_clipping_preserves_direction_without_raw_norm_overflow(magnitude):
    g = {'a': jnp.array([magnitude, -magnitude], dtype=jnp.float32),
         'b': jnp.array([magnitude], dtype=jnp.float32)}
    tx = common.stable_clip_by_global_norm(1.0)
    out, _ = jax.jit(tx.update)(g, tx.init(g))
    flat = np.concatenate([np.asarray(x) for x in jax.tree.leaves(out)]).astype(np.float64)
    assert np.isfinite(flat).all()
    np.testing.assert_allclose(flat, np.array([1., -1., 1.]) / np.sqrt(3), rtol=2e-6)
    assert np.linalg.norm(flat) == pytest.approx(1.0, rel=2e-6)


def test_clipping_zero_and_small_gradients_and_invalid_norm_reporting():
    tx = common.stable_clip_by_global_norm(1.0)
    for g in [jnp.zeros(3), jnp.array([0.2, -0.3])]:
        out, _ = tx.update(g, tx.init(g))
        np.testing.assert_array_equal(out, g)
    assert np.isinf(float(common.safe_norm(jnp.array([jnp.inf, 1.]))))


def run_fake_update(values):
    tx = distill._opt()
    # Constant local derivative; values may deliberately generate a bad loss.
    def loss_one(p, cfg, heads, data):
        value = p['w'] * data['x'].sum()
        return value, {'loss': value, 'act': value}
    fake = SimpleNamespace(N=len(values), n_mb=len(values), passes=1, M=1,
                           cfg={'unused': jnp.zeros(1)}, opt=tx, _loss_one=loss_one,
                           _t_heads=lambda: jnp.zeros(1))
    p = {'w': jnp.array([0.25], dtype=jnp.float32)}
    o = jax.vmap(tx.init)(p)
    data = {'x': jnp.array(values, dtype=jnp.float32)[None, None, :]}
    return jax.jit(lambda p, o: distill.Distiller._update(
        fake, p, o, data, jax.random.PRNGKey(4), jnp.array([1e-3])))(p, o)


def test_finite_huge_gradient_is_accepted_and_adam_state_remains_finite():
    p, o, st, ok = run_fake_update([1e30])
    assert bool(ok[0]) and float(st['grad_rejected'][0]) == 0
    assert bool(common.finite_tree(o))
    assert float(p['w'][0]) < 0.25


def test_bad_minibatch_does_not_erase_good_update_or_poison_optimizer():
    p, o, st, ok = run_fake_update([jnp.nan, 1.0])
    assert bool(ok[0])
    assert float(st['grad_rejected'][0]) == 0.5
    assert float(st['loss_nonfinite'][0]) == 0.5
    assert float(st['grad_nonfinite'][0]) == 0.5
    assert bool(common.finite_tree(o)) and float(p['w'][0]) < 0.25
    assert int(o[1].count[0]) == 1


def test_all_rejected_updates_report_failure_and_preserve_parameters():
    p, o, st, ok = run_fake_update([jnp.nan, jnp.nan])
    assert not bool(ok[0]) and float(st['grad_rejected'][0]) == 1
    assert float(p['w'][0]) == 0.25 and bool(common.finite_tree(o))
    assert int(o[1].count[0]) == 0
