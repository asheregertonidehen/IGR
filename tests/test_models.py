"""Model tests: B0 parity, cell switches, stop-gradient scope, teacher routing, parameter counts, determinism."""
import jax
import jax.numpy as jnp
import numpy as np

from models import baselines, gated_cell, inputs as I, raptor_b0
from models import variants as V

REP_PARAMS = ('rep', 'pc', 'enu', 'gate', 'corr', 'r', 'heads')


def _channels(key, teacher=False):
    ks = jax.random.split(key, 10)
    ch = {c: jax.random.normal(k, (I.CH_DIMS[c],)) for c, k in zip(I.CH_DIMS, ks)}
    for c, i in (('ext_pos', 3), ('ext_vel', 3), ('ref', 9)):
        ch[c] = ch[c].at[i].set(1.0).at[i + 1].set(0.0)        # valid = 1
    if teacher:
        ch['truth'] = jnp.concatenate([jax.random.normal(ks[8], (18,)), jnp.ones(1)])
        ch['descriptor'] = jax.random.normal(ks[9], (I.DESC_DIM,))
    return ch


def _absum(tree):
    return sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree.leaves(tree))


def test_b0_parity_with_foundation_policy():
    from foundation_policy import Raptor
    pol = Raptor()
    pol.reset()
    p = raptor_b0.load_params()
    x = np.random.default_rng(0).normal(size=(30, 2, 22)).astype(np.float32)
    h = jnp.tile(p['h0'], (2, 1))
    for t in range(30):
        ref = np.clip(pol.evaluate_step(x[t]), -1, 1)
        h, a = raptor_b0.step(p, h, jnp.asarray(x[t]))
        assert np.abs(ref - np.asarray(a)).max() < 1e-5


def test_raptor_gru16_param_count():
    assert baselines.gru_param_count(22, 16) + 16 == 2084  # + learned h0 in RAPTOR


def test_use_gates_zero_gives_unit_gates_and_matches_saturated_gates():
    cell = gated_cell.GatedCell('gru')
    p = cell.init(jax.random.PRNGKey(0))
    ch = _channels(jax.random.PRNGKey(1))
    _, mean_a2, aux = cell.step(p, V.single_cfg('A2'), cell.init_carry(), ch)
    assert np.allclose(np.asarray(aux['gates']), 1.0)
    p_sat = dict(p, gate={c: dict(g, w=jnp.zeros_like(g['w']), b=jnp.full(1, 60.0)) for c, g in p['gate'].items()})
    _, mean_full, aux2 = cell.step(p_sat, V.single_cfg('Full'), cell.init_carry(), ch)
    assert np.allclose(np.asarray(aux2['gates']), 1.0)
    assert np.allclose(np.asarray(mean_a2), np.asarray(mean_full), atol=1e-6)


def test_rep_grad_zero_blocks_control_gradient_into_representation_module():
    cell = gated_cell.GatedCell('gru')
    p = cell.init(jax.random.PRNGKey(0))
    seq = [_channels(jax.random.PRNGKey(i)) for i in range(6)]

    def control_loss(params, cfg):
        c, tot = cell.init_carry(), 0.0
        for ch in seq:
            c, mean, _ = cell.step(params, cfg, c, ch)
            tot = tot + jnp.sum(jnp.tanh(mean) ** 2)
        return tot
    g0 = jax.grad(control_loss)(p, V.single_cfg('A4'))
    g1 = jax.grad(control_loss)(p, V.single_cfg('Full'))
    assert _absum({k: g0[k] for k in REP_PARAMS}) == 0.0
    assert _absum({k: g1[k] for k in ('rep', 'pc', 'enu', 'gate', 'corr', 'r')}) > 0.0
    assert _absum(g0['ctl']) > 0.0 and _absum(g0['out']) > 0.0      # the controller still learns


def test_invalid_channel_gives_zero_innovation():
    cell = gated_cell.GatedCell('gru')
    p = cell.init(jax.random.PRNGKey(0))
    ch = _channels(jax.random.PRNGKey(2))
    for c, i in (('ext_pos', 3), ('ext_vel', 3), ('ref', 9)):
        ch[c] = ch[c].at[i].set(0.0)
    _, _, aux = cell.step(p, V.single_cfg('Full'), cell.init_carry(), ch)
    assert np.allclose(np.asarray(aux['nis']), 0.0)
    ch2 = dict(ch, ext_vel=ch['ext_vel'].at[:3].add(5.0), ext_pos=ch['ext_pos'].at[:3].add(-3.0))
    _, _, aux2 = cell.step(p, V.single_cfg('Full'), cell.init_carry(), ch2)
    assert np.allclose(np.asarray(aux['h_rep']), np.asarray(aux2['h_rep']), atol=1e-6)   # memory write unaffected


def test_teacher_representation_never_sees_descriptor_or_truth():
    for prog in ('T',):
        arch = V.teacher_arch(prog)
        p = arch.init(jax.random.PRNGKey(0))
        seq = [_channels(jax.random.PRNGKey(i), teacher=True) for i in range(3)]

        def rep_out(desc, truth, cfg):
            c, tot = arch.init_carry(), 0.0
            for ch in seq:
                c, _, aux = arch.step(p, cfg, c, dict(ch, descriptor=desc, truth=truth))
                tot = tot + jnp.sum(aux['h_rep']) + jnp.sum(aux['pc_mu']) + jnp.sum(aux['truth_head']) + jnp.sum(aux['id_head'])
            return tot
        args = (seq[0]['descriptor'], seq[0]['truth'])
        gd, gt = jax.grad(rep_out, (0, 1))(*args, V.teacher_cfg('T_0'))
        assert float(jnp.sum(jnp.abs(gd))) == 0.0 and float(jnp.sum(jnp.abs(gt))) == 0.0
        # the controller does receive them
        ga = jax.grad(lambda d: jnp.sum(arch.step(p, V.teacher_cfg('T_0'), arch.init_carry(), dict(seq[0], descriptor=d))[1]))(args[0])
        assert float(jnp.sum(jnp.abs(ga))) > 0.0


def test_cells_run_all_backends_and_widths():
    for ct in ('gru', 'lstm', 'mingru'):
        cell = gated_cell.GatedCell(ct)
        p = cell.init(jax.random.PRNGKey(0))
        c, mean, aux = cell.step(p, V.single_cfg('Full'), cell.init_carry(), _channels(jax.random.PRNGKey(2)))
        c, mean, aux = cell.step(p, V.single_cfg('Full'), c, _channels(jax.random.PRNGKey(3)))
        assert mean.shape == (4,) and np.all(np.isfinite(np.asarray(mean)))
    W = gated_cell.widths(3)
    assert (W['h_rep'], W['h_ctl'], W['r']) == (72, 48, 32)


def test_matched_baseline_close_to_full_params():
    full = V.full_deploy_params()
    h = V.matched_hidden()
    assert abs(baselines.gru_param_count(I.NATIVE_DIM, h) - full) / full < 0.05
    assert 9000 < full < 12000
    assert I.NATIVE_DIM == 54 and I.TRUTH_TARGET_DIM == 25


def test_determinism_same_seed_same_first_loss():
    from train import distill
    from models.expert import Expert
    tp = jax.vmap(V.teacher_arch('T').init)(jnp.stack([jax.random.PRNGKey(i) for i in range(4)]))
    e = jax.vmap(Expert().init)(jnp.stack([jax.random.PRNGKey(i) for i in range(3)]))
    seeds = dict(teacher_seeds=[0, 1, 2], student_seeds_core=[0, 1], student_seeds_other=[0], student_seeds_appendix=[0])
    models = [m for m in V.family_models('F2b', seeds) if m[1] in ('B4', 'A7')][:3]
    tn = list(V.TEACHER_PROGRAMS['T'])
    losses = []
    for _ in range(2):
        d = distill.Distiller('F2b', models, 4, tp, tn, e, [0, 1, 2], ep_steps=16, mb_seqs=2)
        params, opt = d.init_params(jax.random.PRNGKey(3))
        data, _ = d.rollout(params, jax.random.PRNGKey(4), 1.0, False)
        _, _, st, _ = d.update(params, opt, data, jax.random.PRNGKey(5), jnp.full(d.M, 1e-3))
        losses.append(np.asarray(st['loss']))
    assert np.array_equal(losses[0], losses[1])


def test_gated_distillation_step_is_finite():
    from train import distill
    from models.expert import Expert
    tp = jax.vmap(V.teacher_arch('T').init)(jnp.stack([jax.random.PRNGKey(i) for i in range(4)]))
    e = jax.vmap(Expert().init)(jnp.stack([jax.random.PRNGKey(i) for i in range(3)]))
    seeds = dict(teacher_seeds=[0], student_seeds_core=[0, 1], student_seeds_other=[0], student_seeds_appendix=[0])
    models = V.family_models('F1_S2', seeds)
    d = distill.Distiller('F1_S2', models, 2, tp, list(V.TEACHER_PROGRAMS['T']), e, [0, 1, 2], ep_steps=12, mb_seqs=1,
                          lam_star=0.1)
    params, opt = d.init_params(jax.random.PRNGKey(0))
    data, rst = d.rollout(params, jax.random.PRNGKey(1), 1.0, True)
    p2, _, st, ok = d.update(params, opt, data, jax.random.PRNGKey(2), jnp.full(d.M, 1e-3))
    assert bool(np.all(np.asarray(ok))) and np.all(np.isfinite(np.asarray(st['loss'])))
    dg = d.diag(params, data)
    assert np.all(np.isfinite(np.asarray(dg['cos_truth'])))
    a3 = [i for i, m in enumerate(models) if m[1] == 'A3'][0]
    assert float(st['loss'][a3]) == float(st['act'][a3])        # lambda = 0 everywhere


def test_target_standardization_roundtrip():
    st = I.target_stats()
    m, s = st['truth']
    y = jnp.linspace(-1, 1, I.TRUTH_TARGET_DIM)
    assert np.allclose((I.truth_destandardize(y) - m) / s, y, atol=1e-6)
    assert all(float(jnp.min(v[1])) > 0 for v in st.values())
    assert set(st) == {'truth', 'id', 'pred', 'pc'}


def test_resampling_probabilities_floor_and_order():
    from train import ppo
    score = jnp.array([1.0, 0.5, 0.0, 0.2])
    p = np.asarray(ppo.resample_probs(score, jnp.ones(4, bool)))
    assert abs(p.sum() - 1) < 1e-6 and p.min() >= 0.25 / 4 - 1e-7
    assert p[2] > p[1] > p[0]      # harder drones (lower return) are sampled more


def test_variant_table():
    seeds = dict(teacher_seeds=[0, 1, 2], student_seeds_core=[0, 1], student_seeds_other=[0])
    all_models = [m for f in V.FAMILIES for m in V.family_models(f, seeds)]
    count = lambda v: sum(1 for m in all_models if m[1] == v)
    assert count('Full') == 3 and len(V.family_models('F1_S1', seeds)) == 3      # + 3 lambda_s models from F1_S1 = 6
    for v in ('A1', 'A3', 'A7', 'A9', 'B4', 'B5m'):
        assert count(v) == 6, v
    for v in ('A2', 'A3a', 'A3b', 'A3c', 'A4', 'A5', 'A11'):
        assert count(v) == 3, v
    assert list(V.TEACHER_PROGRAMS) == ['T'] and V.TEACHER_PROGRAMS['T'] == ['T_0', 'T_1', 'T_2']
    assert V.parse_model('Full_l0.03_t2_s1') == ('Full_l0.03', 2, 1)
    assert float(V.stack_cfg(['A2'], lam_star=0.03)['lam_scale'][0]) == np.float32(0.03)
    assert float(V.stack_cfg(['Full'])['corr_scale'][0]) == 0.5 and float(V.teacher_cfg('T_0')['corr_scale']) == 1.0
    assert float(V.stack_cfg(['B4'])['corr_scale'][0]) == 1.0
    assert V.VARIANTS['Full']['lam_g'] == 0.1


def test_innovation_gradient_is_cut_only_when_pc_is_supervised():
    """With lam_pc > 0 the control loss sends no gradient into P_c through the innovation;
    with lam_pc = 0 (A3, A3c) the innovation stays free. The forward value is identical."""
    cell = gated_cell.GatedCell('gru')
    p = cell.init(jax.random.PRNGKey(0))
    ch = _channels(jax.random.PRNGKey(1))

    def loss(params, cfg):
        c, tot = cell.init_carry(), 0.0
        for _ in range(4):
            c, mean, _ = cell.step(params, cfg, c, ch)
            tot = tot + jnp.sum(jnp.tanh(mean) ** 2)
        return tot
    g_full = jax.grad(loss)(p, V.single_cfg('Full'))
    g_free = jax.grad(loss)(p, V.single_cfg('A3c'))
    assert _absum(g_full['pc']) == 0.0 and _absum(g_free['pc']) > 0.0
    assert _absum(g_full['enu']) > 0.0 and _absum(g_full['gate']) > 0.0      # the rest of the module still learns
    a = cell.step(p, V.single_cfg('Full'), cell.init_carry(), ch)[1]
    b = cell.step(p, V.single_cfg('Full', lam_pc=0.0), cell.init_carry(), ch)[1]
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_truncated_bptt_matches_full_loss_value_and_stays_finite():
    """100-step windows with carried state give the same forward loss as full BPTT; gradients stay finite."""
    from train import distill
    from models.expert import Expert
    tp = jax.vmap(V.teacher_arch('T').init)(jnp.stack([jax.random.PRNGKey(i) for i in range(4)]))
    e = jax.vmap(Expert().init)(jnp.stack([jax.random.PRNGKey(i) for i in range(3)]))
    seeds = dict(teacher_seeds=[0], student_seeds_core=[0, 1], student_seeds_other=[0], student_seeds_appendix=[0])
    models = [m for m in V.family_models('F1_S2', seeds) if m[1] == 'Full'][:1]
    out = []
    for bptt in (None, 4):
        d = distill.Distiller('F1_S2', models, 2, tp, list(V.TEACHER_PROGRAMS['T']), e, [0, 1, 2], ep_steps=12, mb_seqs=1,
                              lam_star=0.1, bptt=bptt)
        params, opt = d.init_params(jax.random.PRNGKey(0))
        data, _ = d.rollout(params, jax.random.PRNGKey(1), 1.0, True)
        _, _, st, ok = d.update(params, opt, data, jax.random.PRNGKey(2), jnp.full(d.M, 1e-3))
        assert bool(np.all(np.asarray(ok))) and float(st['grad_rejected'][0]) == 0.0
        # forward loss at the initial parameters (no update applied)
        parts, _ = d._losses(jax.tree.map(lambda x: x[0], params), jax.tree.map(lambda x: x[0], d.cfg),
                             jax.tree.map(lambda x: x[0], d._t_heads()), jax.tree.map(lambda x: x[0][:, :1], data))
        out.append(float(parts['act']))
    assert abs(out[0] - out[1]) < 1e-6, out
