"""Compute the reported numbers from a finished run (results/, logs/). Run from the repository root.
Writes results/analysis/numbers.json. Intervals are 95 % t-intervals over the 3 teacher-seed means (df = 2)."""
import glob, json, os, re
import numpy as np, pandas as pd, scipy.stats as st

R = 'results'
os.makedirs('results/analysis', exist_ok=True)
SENS = ['delay_60', 'delay_100', 'delay_150', 'delay_burst', 'reorder_severe', 'ref_late', 'freeze_long', 'low_rate',
        'dropout_sustained', 'dark', 'imu_bias_3x', 'vibration_3x']
FAM = dict(S=SENS, A=['act_delay_40', 'act_delay_60'], D=['wind_strong', 'payload_40', 'motor_loss', 'high_t2w'],
           K=['fast_traj', 'figure8_fast', 'midair'], C=['combo_delay_gust', 'combo_all'])
HELD = sum(FAM.values(), [])
assert len(HELD) == 23 and len(SENS) == 12


def parse(m):
    x = re.match(r'(.+?)_t(\d)_s(\d)$', m)
    if not x:
        return m, -1, -1
    v = 'Full' if x.group(1).startswith('Full') else x.group(1)
    return v, int(x.group(2)), int(x.group(3))


fr = []
for g in ('F1_S1', 'F1_S2', 'F2b', 'F2b_A9', 'T', 'E', 'B0'):
    for f in glob.glob(f'{R}/eval/{g}/*.parquet'):
        fr.append(pd.read_parquet(f, columns=['success', 'rmse_clipped', 'zosc_power', 'alive5', 'model', 'suite', 'episode', 't2w']))
D = pd.concat(fr, ignore_index=True)
P = {m: parse(m) for m in D.model.unique()}
for i, c in enumerate(('variant', 't', 's')):
    D[c] = D.model.map(lambda m: P[m][i])
S = D.groupby(['variant', 't', 's', 'suite'], observed=True).agg(success=('success', 'mean'), rmse=('rmse_clipped', 'mean'),
                                                                  zosc=('zosc_power', 'mean')).reset_index()


def student_value(v, suites, metric, s0=False):
    x = S[(S.variant == v) & S.suite.isin(suites)]
    if s0:
        x = x[x.s == 0]
    return x.groupby(['t', 's'])[metric].mean()


def contrast(a, b, suites, metric='success', s0=False, scale=100.0):
    """Teacher-level contrast: difference of the teacher means of student values."""
    x = student_value(a, suites, metric, s0).groupby('t').mean()
    y = student_value(b, suites, metric, s0).groupby('t').mean()
    d = (x - y) * scale
    m, h = d.mean(), st.t.ppf(.975, 2) * d.std(ddof=1) / np.sqrt(3)
    nd = 5
    return dict(diff=round(float(m), nd), lo=round(float(m - h), nd), hi=round(float(m + h), nd), seeds=[round(float(z), nd) for z in d])


def mean_of(v, suites, metric='success', scale=100.0):
    return round(float(student_value(v, suites, metric).mean()) * scale, 5)


N = {}
VS = ['B0', 'B4', 'B5m', 'A7', 'A9', 'A1', 'A2', 'A3', 'A3a', 'A3b', 'A3c', 'A4', 'A5', 'A11', 'Full', 'T_0', 'T_1', 'T_2', 'E_0', 'E_1', 'E_2']
tab = {}
for v in VS:
    r = {k: mean_of(v, s) for k, s in FAM.items()}
    r.update(heldout=mean_of(v, HELD), val_id=mean_of(v, ['val_id']), test_id=mean_of(v, ['test_id']), clean=mean_of(v, ['clean']))
    r['n'] = int(S[S.variant == v].groupby(['t', 's']).ngroups)
    tab[v] = r
tab['T'] = {k: round(float(np.mean([tab[t][k] for t in ('T_0', 'T_1', 'T_2')])), 5) for k in ('S', 'A', 'D', 'K', 'C', 'heldout', 'val_id', 'test_id', 'clean')}
N['table'] = tab
SU = HELD + ['val_id', 'test_id', 'clean', 'delay_0', 'delay_20', 'delay_40', 'delay_80']
N['suites'] = {su: {v: mean_of(v, [su]) for v in VS} for su in SU}
N['rmse_all_episodes'] = {su: {v: mean_of(v, [su], 'rmse', 1.0) for v in ('A3c', 'A7', 'A9', 'A3', 'Full', 'B5m')} for su in ('low_rate', 'dark', 'dropout_sustained')}
N['rmse_all_episodes']['heldout'] = {v: mean_of(v, HELD, 'rmse', 1.0) for v in ('A3c', 'A7', 'A9', 'A3', 'Full', 'B5m')}
C = {}
for name, (a, b, su, kw) in {
    'A3c-A7 heldout': ('A3c', 'A7', HELD, {}), 'A3c-A7 S': ('A3c', 'A7', SENS, {}),
    'A3c-Full heldout': ('A3c', 'Full', HELD, {}), 'A3c-Full S': ('A3c', 'Full', SENS, {}),
    'A3c-Full heldout s0': ('A3c', 'Full', HELD, dict(s0=True)), 'A3c-Full S s0': ('A3c', 'Full', SENS, dict(s0=True)),
    'A3c-A3 heldout s0': ('A3c', 'A3', HELD, dict(s0=True)), 'A3c-A3 S s0': ('A3c', 'A3', SENS, dict(s0=True)),
    'A3c-A9 heldout': ('A3c', 'A9', HELD, {}), 'A3c-A9 S': ('A3c', 'A9', SENS, {}),
    'A3-Full heldout': ('A3', 'Full', HELD, {}), 'Full-A7 heldout': ('Full', 'A7', HELD, {}), 'Full-A7 S': ('Full', 'A7', SENS, {}),
    'Full-B4 S': ('Full', 'B4', SENS, {}), 'Full-B5m S': ('Full', 'B5m', SENS, {}), 'Full-A1 heldout': ('Full', 'A1', HELD, {}),
    'A7-B4 heldout': ('A7', 'B4', HELD, {}), 'A7-B4 S': ('A7', 'B4', SENS, {}), 'A7-B4 val_id': ('A7', 'B4', ['val_id'], {}),
    'A3c-B4 heldout': ('A3c', 'B4', HELD, {}), 'A3c-B4 S': ('A3c', 'B4', SENS, {}), 'A3c-B5m S': ('A3c', 'B5m', SENS, {}), 'A3c-A7 val_id': ('A3c', 'A7', ['val_id'], {}),
    'A3c-A7 low': ('A3c', 'A7', ['low_rate'], {}), 'A3c-A9 low': ('A3c', 'A9', ['low_rate'], {}), 'A3c-A3 low s0': ('A3c', 'A3', ['low_rate'], dict(s0=True)),
    'A3c-Full fig8': ('A3c', 'Full', ['figure8_fast'], {}), 'A3c-Full fast': ('A3c', 'Full', ['fast_traj'], {}),
    'A3c-A7 drop': ('A3c', 'A7', ['dropout_sustained'], {}), 'A3c-A7 dark': ('A3c', 'A7', ['dark'], {}),
    'A3c-A7 outage': ('A3c', 'A7', ['dark', 'dropout_sustained'], {}), 'Full-A7 outage': ('Full', 'A7', ['dark', 'dropout_sustained'], {}),
    'A1-B4 outage': ('A1', 'B4', ['dark', 'dropout_sustained'], {}), 'A3-A7 outage': ('A3', 'A7', ['dark', 'dropout_sustained'], {}),
    'A3c-A9 outage': ('A3c', 'A9', ['dark', 'dropout_sustained'], {}),
    'Full-A2 outage': ('Full', 'A2', ['dark', 'dropout_sustained'], {}), 'Full-A5 outage': ('Full', 'A5', ['dark', 'dropout_sustained'], {}),
    'Full-A11 outage': ('Full', 'A11', ['dark', 'dropout_sustained'], {}),
}.items():
    C[name] = contrast(a, b, su, **kw)
C['A3c-A7 low rmse'] = contrast('A3c', 'A7', ['low_rate'], 'rmse', scale=1.0)
C['A3c-A9 low rmse'] = contrast('A3c', 'A9', ['low_rate'], 'rmse', scale=1.0)
C['A3c-A7 heldout rmse'] = contrast('A3c', 'A7', HELD, 'rmse', scale=1.0)
C['A3c-A7 low zosc'] = contrast('A3c', 'A7', ['low_rate'], 'zosc', scale=1.0)
N['low_zosc'] = {v: round(float(student_value(v, ['low_rate'], 'zosc').mean()), 5) for v in ('A3c', 'A7')}
N['contrasts'] = C
N['ablation_heldout'] = {v: contrast(v, 'Full', HELD) for v in ('A1', 'A2', 'A3', 'A3a', 'A3b', 'A3c', 'A4', 'A5', 'A11', 'A7', 'A9')}
N['ablation_S'] = {v: contrast(v, 'Full', SENS) for v in ('A1', 'A2', 'A3', 'A3a', 'A3b', 'A3c', 'A4', 'A5', 'A11', 'A7', 'A9')}
N['wins'] = {f'{a} vs {b}': dict(S=int(sum(N['suites'][s][a] > N['suites'][s][b] for s in SENS)), held=int(sum(N['suites'][s][a] > N['suites'][s][b] for s in HELD)))
             for a, b in (('A7', 'B5m'), ('A3c', 'B5m'), ('Full', 'A7'), ('Full', 'B4'), ('A3c', 'A7'), ('A3c', 'Full'))}
s_noout = [s for s in SENS if s not in ('dark', 'dropout_sustained')]
N['S_no_outage'] = {v: mean_of(v, s_noout) for v in ('A7', 'B5m', 'A3c', 'Full', 'B4')}
# delay at 50 % success
ds = [('delay_0', 0), ('delay_20', 20), ('delay_40', 40), ('delay_60', 60), ('delay_80', 80), ('delay_100', 100), ('delay_150', 150)]
d50 = {}
for v in ('B0', 'B4', 'B5m', 'A7', 'A3c', 'Full', 'A1'):
    y = [N['suites'][s][v] for s, _ in ds]; x = [d for _, d in ds]
    k = [i for i in range(6) if y[i] >= 50 > y[i + 1]]
    d50[v] = round(x[k[0]] + (y[k[0]] - 50) / (y[k[0]] - y[k[0] + 1]) * (x[k[0] + 1] - x[k[0]]), 1) if k else None
    d50[v + '_curve'] = y
N['d50'] = d50
# paired-episode complementarity A7 / B5m
rows = []
for su in HELD:
    for t in (0, 1, 2):
        for ss in (0, 1):
            a = D[(D.variant == 'A7') & (D.t == t) & (D.s == ss) & (D.suite == su)].sort_values('episode').success.values
            b = D[(D.variant == 'B5m') & (D.t == t) & (D.s == ss) & (D.suite == su)].sort_values('episode').success.values
            rows.append(dict(suite=su, u=np.maximum(a, b).mean()))
N['oracle_union_A7_B5m'] = round(float(pd.DataFrame(rows).groupby('suite').u.mean().mean() * 100), 2)
N['oracle_suite_A7_B5m'] = round(float(np.mean([max(N['suites'][s]['A7'], N['suites'][s]['B5m']) for s in HELD])), 2)
# probes
pr = {}
for g in ('F1_S1', 'F1_S2', 'F2b', 'F2b_A9'):
    pr.update({k: v for k, v in json.load(open(f'{R}/probes/{g}.json')).items()})
N['_probe_keys'] = list(pr)[:3]
json.dump(pr, open('results/analysis/_probes_raw.json', 'w'))
# events
ev = {}
for prot in ('primary', 'secondary'):
    ev[prot] = {}
    for g in ('F1_S1', 'F1_S2', 'F2b', 'T'):
        for e, d in json.load(open(f'{R}/discrimination/{g}_{prot}.json')).items():
            for m, r in d.items():
                ev[prot].setdefault(e, {}).setdefault(parse(m)[0], []).append(r)
evs = {}
for prot, E in ev.items():
    evs[prot] = {}
    for e, d in E.items():
        evs[prot][e] = {}
        for v, rs in d.items():
            evs[prot][e][v] = {k: round(float(np.nanmean([r.get(k, np.nan) for r in rs])), 4) for k in
                               ('X', 'event_crashes', 'n_valid', 'detect_frac', 'mcont_rep', 'gate_ref_pre', 'gate_ref_event', 'gate_ext_vel_pre', 'gate_ext_vel_event',
                                'nis_ext_vel_pre', 'nis_ext_vel_event', 'nis_ref_pre', 'nis_ref_event')}
            evs[prot][e][v]['n_students'] = len(rs)
N['events'] = evs
# gradients
gr = {}
for name, path in (('follow', 'logs/metrics/F1_S1.jsonl'), ('follow', 'logs/metrics/F1_S2.jsonl'), ('follow', 'logs/metrics/F2b_A9.jsonl'), ('plain', 'logs/metrics/F2b.jsonl')):
    rows_ = {}
    for l in open(path):                     # a resumed or restarted log repeats epochs: the last row wins
        r = json.loads(l)
        rows_[(r['tag'], r['epoch'])] = r
    for r in rows_.values():
        for m, x in r['models'].items():
            k = f'{name}:{m}'
            gr.setdefault(k, []).append((r['epoch'], x.get('grad_norm', np.nan), x.get('grad_norm_max', np.nan), x.get('grad_rejected', 0.0), x.get('act', np.nan)))
g2 = {}
for k, a in gr.items():
    a = np.array(a, float); late = a[a[:, 0] >= 400]
    g2[k] = dict(n=len(a), peak_all=float(np.nanmax(a[:, 2])), peak_400=float(np.nanmax(late[:, 2])) if len(late) else None,
                 med_400=float(np.nanmedian(late[:, 1])) if len(late) else None, rej=float(16 * np.nansum(a[:, 3])))
N['grad'] = g2
json.dump({k: np.array(v, float)[:, :3].tolist() for k, v in gr.items() if 'Full' in k or 'A3c' in k}, open('results/analysis/_grad_curves.json', 'w'))
json.dump(N, open('results/analysis/numbers.json', 'w'), indent=1, default=str)
