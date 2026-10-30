"""Paired-event metrics and the convergence plateau rule on synthetic data."""
import numpy as np

from eval import jobs

KEEP = dict(post_window=2.0, pre_window=2.0, detect={'E2': ['force', 0.5], 'E7': ['t2w', 1.0]})


def _rec(n=4, T=1200, M=1):
    z = lambda: np.zeros((M, n, T))
    rec = dict(perr_ev=z() + 0.1, perr_cf=z() + 0.1, alive_ev=np.ones((M, n, T)), alive_cf=np.ones((M, n, T)),
               adev=z(), mcont=z(), mcont_rep=z(), S_v=z() + 1.0, force_head=z() + 0.2, t2w_head=z() + 3.0,
               vel_head_err=z(), gates=np.ones((M, n, T, 3)), nis=np.ones((M, n, T, 3)))
    rng = np.random.default_rng(0)
    rec['force_head'] += 0.001 * rng.normal(size=rec['force_head'].shape)
    rec['t2w_head'] += 0.001 * rng.normal(size=rec['t2w_head'].shape)
    return rec


def test_tracking_excess_and_crash_counted_as_maximal_harm():
    rec = _rec()
    t_ev = np.full(4, 7.0)
    k0 = 700
    rec['perr_ev'][0, :, k0:k0 + 100] += 0.5            # 1 s of +0.5 m excess -> X = 0.5 m s
    rec['alive_ev'][0, 3, k0 + 50:] = 0                  # drone 3 crashes 0.5 s after onset
    out = jobs.h2_metrics(rec, 0, t_ev, 0.5, KEEP, np.full(4, 2.0), 'E1')
    assert out['n_valid'] == 4 and out['event_crashes'] == 1
    x_crash = (2.0 - 0.1) * (k0 + 250 - (k0 + 50) + 1) * 0.01      # threshold until window end (0.5 + 2 s)
    expect = (3 * 0.5 + 0.5 * 0.5 + x_crash) / 4
    assert abs(out['X'] - expect) < 0.02, (out['X'], expect)


def test_counterfactual_survival_filters_drones():
    rec = _rec()
    rec['alive_cf'][0, 0, 800:] = 0                       # the counterfactual itself crashes in the window
    out = jobs.h2_metrics(rec, 0, np.full(4, 7.0), 0.5, KEEP, np.full(4, 2.0), 'E1')
    assert out['n_valid'] == 3


def test_head_detection_within_window():
    rec = _rec()
    rec['force_head'][0, :2, 720:] += 0.1                 # rises 0.2 s after onset in 2 of 4 drones
    out = jobs.h2_metrics(rec, 0, np.full(4, 7.0), 0.6, KEEP, np.full(4, 2.0), 'E2')
    assert abs(out['detect_frac'] - 0.5) < 1e-9
    rec = _rec()
    rec['t2w_head'][0, :, 760:] -= 0.5                    # payload lowers effective T/W within 1 s
    out = jobs.h2_metrics(rec, 0, np.full(4, 7.0), 1.0, KEEP, np.full(4, 2.0), 'E7')
    assert out['detect_frac'] == 1.0


def test_plateau_rule():
    cfg = dict(window_frac=0.2, success_pp=2, rmse_pct=5)
    flat = {str(e): [0.90, 0.15] for e in range(100, 1501, 100)}
    assert jobs.plateau(flat, 1500, cfg)[0]
    late_gain = dict(flat, **{'1400': [0.93, 0.15]})
    assert not jobs.plateau(late_gain, 1500, cfg)[0]
    rmse_gain = dict(flat, **{'1500': [0.90, 0.13]})
    assert not jobs.plateau(rmse_gain, 1500, cfg)[0]


def test_variant_curves_average_students(tmp_path, monkeypatch):
    import json
    from train import common
    monkeypatch.setattr(common, 'RESULTS', tmp_path)
    (tmp_path / 'select_ckpt_FX').mkdir()
    curves = {'Full_t0_s0': {'100': [0.8, 0.1], '200': [0.9, 0.1]}, 'Full_t1_s0': {'100': [0.6, 0.3], '200': [0.7, 0.3]}}
    (tmp_path / 'select_ckpt_FX' / 'meta.json').write_text(json.dumps(dict(curves=curves)))
    out = jobs.variant_curves(['FX'])
    assert list(out) == ['Full'] and abs(out['Full']['100'][0] - 0.7) < 1e-9 and abs(out['Full']['200'][1] - 0.2) < 1e-9
