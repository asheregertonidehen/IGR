"""Simulator and sensing behaviour: action history, effective identity, event flags and reorder bursts."""
import jax
import jax.numpy as jnp
import numpy as np

from sim import env as envm
from sim import sampler, specs, tasks
from sensing import links


def _dset(n=16, seed=5):
    return {k: jnp.asarray(v) for k, v in sampler.sample_drones(n, seed).items()}


def _hold_spec(**kw):
    spec, _ = specs.train_spec()
    spec = dict(spec, task_probs=jnp.array([1.0, 0, 0, 0, 0]), init_at_target=jnp.asarray(1.0),
                mix=jnp.array([1.0, 0, 0, 0]))
    spec.update({k: jnp.asarray(v, jnp.float32) for k, v in kw.items()})
    return spec


def test_action_history_k6_in_obs():
    spec, _ = specs.train_spec()
    st = envm.reset(envm.sample_episode(jax.random.PRNGKey(0), _dset(), spec, 1.0))
    assert st['obs']['prev_actions'].shape == (24,)
    acts = [jnp.full(4, 0.1 * (i + 1)) for i in range(7)]
    for i, a in enumerate(acts):
        st, _, _ = envm.step(st, a, jax.random.PRNGKey(i))
    h = np.asarray(st['obs']['prev_actions']).reshape(6, 4)
    assert np.allclose(h[:, 0], [0.7, 0.6, 0.5, 0.4, 0.3, 0.2], atol=1e-6)     # most recent first


def test_effective_identity_payload_scales_t2w():
    """A scripted +30 % payload moves the r_t2w target by 1/1.3 (and log r_t2i by -log 1.3)."""
    spec = _hold_spec()
    ep = envm.sample_episode(jax.random.PRNGKey(1), _dset(), spec, 0.0, script=dict(kind=jnp.asarray(7), t=jnp.asarray(0.05)))
    st = envm.reset(ep)
    id0 = np.asarray(envm.privileged(st)['identity'])
    hover = jnp.full(4, 0.0)
    for i in range(10):
        st, _, _ = envm.step(st, hover, jax.random.PRNGKey(i))
    id1 = np.asarray(envm.privileged(st)['identity'])
    assert np.isclose(id1[0], id0[0] / 1.3, rtol=1e-4)
    assert np.isclose(id1[1], id0[1] - np.log(1.3), atol=1e-4)
    assert np.allclose(id1[2:], id0[2:])


def test_event_active_flags_freeze_and_tail():
    spec = _hold_spec()
    dset = dict(_dset(), sigma_fext=jnp.zeros(16))       # no constant external force: open-loop hover stays alive
    ep = envm.sample_episode(jax.random.PRNGKey(2), dset, spec, 0.0, script=dict(kind=jnp.asarray(1), t=jnp.asarray(0.3)))
    st = envm.reset(ep)
    flags = []
    step = jax.jit(envm.step)
    hover = 2 * ep['init'][4] - 1          # initial motor state is the hover setpoint
    for i in range(160):
        st, _, _ = step(st, hover, jax.random.PRNGKey(i))
        assert bool(st['alive'])
        flags.append(float(envm.privileged(st)['event_active']))
    t = (np.arange(160) + 1) * 0.01
    f = np.array(flags)
    assert f[t < 0.29].max() == 0.0            # before the E1 freeze (0.3 s)
    assert f[(t > 0.31) & (t < 0.79)].min() == 1.0   # freeze 0.5 s
    assert f[(t > 0.81) & (t < 1.29)].min() == 1.0   # + 0.5 s tail
    assert f[t > 1.31].max() == 0.0


def test_maneuver_placed_freezes_start_at_peak_reference_acceleration():
    spec, _ = specs.train_spec()
    spec = dict(spec, task_probs=jnp.array([0.0, 0, 0, 0, 1.0]), ext_freeze_prob=jnp.asarray(1.0),
                ext_freeze_peak_frac=jnp.asarray(1.0), mix=jnp.array([0.0, 1.0, 0, 0]))
    ep = envm.sample_episode(jax.random.PRNGKey(4), _dset(), spec, 1.0)
    peak = tasks.peak_accel_time(ep['task'], spec['event_t'][0], spec['event_t'][1])
    assert np.isclose(float(ep['sens']['vel']['freeze_t']), float(peak))
    grid = np.linspace(0.5, 4.5, 400)
    acc = np.asarray(jax.vmap(lambda s: tasks.ref_accel(ep['task'], s))(grid))
    assert abs(float(peak) - grid[np.argmax((acc ** 2).sum(1))]) < 0.05
    # analytic reference acceleration = derivative of the reference velocity
    _, va = tasks.ref_state(ep['task'], 2.0)
    _, vb = tasks.ref_state(ep['task'], 2.0 + 1e-3)
    assert np.allclose((np.asarray(vb) - np.asarray(va)) / 1e-3, np.asarray(tasks.ref_accel(ep['task'], 2.0)), atol=5e-2)


def test_moderate_langevin_is_faster():
    spec, _ = specs.train_spec()
    sp = lambda kind: [float(np.linalg.norm(np.asarray(tasks.ref_state(
        tasks.sample_task(jax.random.PRNGKey(i), dict(spec, task_probs=jnp.eye(5)[kind])), 3.0)[1]))) for i in range(64)]
    assert np.mean(sp(tasks.MODERATE)) > 1.3 * np.mean(sp(tasks.LANGEVIN))


def test_arrival_stream_fixed_delay():
    prm = {k: jnp.asarray(v, jnp.float32) for k, v in dict(
        rate=100.0, d0=0.03, sigma_j=0.0, p_drop=0.0, freeze_t=1e9, freeze_dur=0.0, burst_t=1e9, burst_dur=0.0,
        burst_delay=0.0, burst_window=0.0, outage_t=1e9, outage_dur=0.0).items()}
    st = links.init(jnp.zeros(1), 1.0)
    for k in range(1, 20):
        t = jnp.asarray(k * 0.01, jnp.float32)
        st, out, _, _, arr = links.step(st, prm, t, jnp.array([float(k)]), jnp.asarray([1.0, 0.0, 0.0]))
        if k > 5:
            m = np.asarray(arr['mask']) > 0
            assert m.sum() == 1 and float(arr['val'][m][0, 0]) == k - 3 == float(out[0])
            assert np.isclose(float(arr['sent'][m][0]), (k - 3) * 0.01, atol=1e-5)


def test_counterfactual_identity_no_event_zero_contamination():
    from eval import jobs, modelsets
    from models import variants as V
    fam = 'F1_S2'
    arch = modelsets.fam_arch(fam)
    seeds = dict(teacher_seeds=[0], student_seeds_core=[0, 1], student_seeds_other=[0], student_seeds_appendix=[0])
    models = [m for m in V.family_models(fam, seeds) if m[1] in ('Full', 'A2')]
    params = jax.vmap(arch.init)(jnp.stack([jax.random.PRNGKey(i) for i in range(len(models))]))
    ms = modelsets.family_set(fam, params, models, lam=0.1, tuning={})
    mc, ad = jobs.counterfactual_identity(ms, n=2, steps=40)
    assert mc == 0.0 and ad == 0.0
