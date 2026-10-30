"""Add probe, stability, event and reliance summaries to results/analysis/numbers.json."""
import json, re, numpy as np, scipy.stats as st
N = json.load(open('results/analysis/numbers.json'))
P = json.load(open('results/analysis/_probes_raw.json'))
def parse(m):
    x = re.match(r'(.+?)_t(\d)_s(\d)$', m); v = 'Full' if x.group(1).startswith('Full') else x.group(1)
    return v, int(x.group(2)), int(x.group(3))
print(list(P)[:3], type(P[list(P)[0]]))
R = 'results'
pr = {}
for g in ('F1_S1', 'F1_S2', 'F2b', 'F2b_A9'):
    for su, d in json.load(open(f'{R}/probes/{g}.json')).items():
        for m, r in d.items():
            v, t, s = parse(m); pr.setdefault(su, {}).setdefault(v, {})[(t, s)] = r
def pm(su, v, k, s0=False):
    a = {ts: r[k] for ts, r in pr[su][v].items() if (not s0 or ts[1] == 0)}
    return np.mean(list(a.values())), {t: np.mean([x for (tt, s), x in a.items() if tt == t]) for t in (0, 1, 2)}
out = {}
for su in ('delay_100', 'val_id'):
    for k in ('v_true_pca16', 'f_ext_pca16', 't2w_pca16', 'v_true', 'f_ext', 't2w'):
        out[f'{su}:{k}'] = {v: round(pm(su, v, k)[0], 3) for v in ('A3c', 'A3', 'A7', 'A9', 'Full', 'B4', 'B5m', 'A2', 'A5', 'A3b', 'A1')}
        out[f'{su}:{k}:s0'] = {v: round(pm(su, v, k, True)[0], 3) for v in ('A3c', 'A3', 'A7', 'A9', 'Full')}
    for a, b in (('A3c', 'A3'), ('A3c', 'A7'), ('A3c', 'A9')):
        for k in ('v_true_pca16', 'f_ext_pca16'):
            x, y = pm(su, a, k, True)[1], pm(su, b, k, True)[1]; d = [x[t] - y[t] for t in (0, 1, 2)]
            out[f'{su}:{k}:{a}-{b}:seeds'] = [round(z, 3) for z in d]
for k, v in out.items(): print(k, v)
N['probes'] = out
rows = {}
for k, v in N['grad'].items():
    run, m = k.split(':'); vv = parse(m)[0]
    rows.setdefault((run, vv), []).append((v['peak_all'], v['peak_400'], v['med_400'], v['rej']))
g = {}
for (run, vv), a in sorted(rows.items()):
    a = np.array(a, float); g[f'{run}:{vv}'] = dict(n=len(a), peak_all_max=float(a[:, 0].max()), peak_400_max=float(a[:, 1].max()), peak_400_min=float(a[:, 1].min()), med=float(np.median(a[:, 2])), rej=float(a[:, 3].sum()))
    print(run, vv, len(a), f"peak all {a[:,0].max():.3g}  peak>=400 max {a[:,1].max():.3g} min {a[:,1].min():.3g} med {np.median(a[:,2]):.3g} rej {a[:,3].sum():.0f}")
N['grad_variant'] = g
E = N['events']
for prot in ('primary', 'secondary'):
    print(prot)
    for e in ('E1', 'E4', 'E5', 'E2', 'E7'):
        print(e, {v: (E[prot][e][v]['X'], E[prot][e][v]['detect_frac'], E[prot][e][v]['event_crashes'], E[prot][e][v]['n_valid']) for v in ('B5m', 'A7', 'A2', 'A3', 'Full', 'A3c') if v in E[prot][e]})
print({v: (E['primary']['E5'][v]['gate_ref_pre'], E['primary']['E5'][v]['gate_ref_event'], E['primary']['E5'][v]['nis_ref_pre'], E['primary']['E5'][v]['nis_ref_event']) for v in ('Full', 'A3c')})
print({e: {v: (E['primary'][e][v]['gate_ext_vel_pre'], E['primary'][e][v]['gate_ext_vel_event'], E['primary'][e][v]['nis_ext_vel_pre'], E['primary'][e][v]['nis_ext_vel_event']) for v in ('Full', 'A3c')} for e in ('E1', 'E4')})
rel = {}
for g2 in ('F1_S1', 'F1_S2', 'F2b'):
    d = json.load(open(f'{R}/reliance/{g2}.json'))
    for su, x in d.items():
        for m, r in x.items():
            v = parse(m)[0]; rel.setdefault(su, {}).setdefault(v, []).append((r.get('d_success_r_mean'), r.get('d_success_r_noise')))
print({su: {v: (round(float(np.mean([a[0] for a in x if a[0] is not None])), 3) if any(a[0] is not None for a in x) else None, len(x)) for v, x in d.items()} for su, d in rel.items()})
N['reliance'] = {su: {v: float(np.mean([a[0] for a in x if a[0] is not None])) for v, x in d.items() if any(a[0] is not None for a in x)} for su, d in rel.items()}
import glob
W = dict(E1=.5, E2=.6, E4=1., E5=1., E7=1.)
H = {}
for f in glob.glob(f'{R}/discrimination/*_primary_E*.npz'):
    x = re.match(r'.*/\w+?_primary_(E\d)_(.+)\.npz', f); ev, m = x.groups()
    if ev not in W or m.startswith('T_'):
        continue
    v = re.sub(r'_t\d_s\d$', '', m); v = 'Full' if v.startswith('Full') else v
    if v not in ('A3c', 'Full', 'A3'):
        continue
    z = np.load(f); t = z['t']; pre = (t > -2) & (t < 0); w = (t >= 0) & (t < W[ev])
    for k in ('force_head', 't2w_head'):
        H.setdefault(v, {}).setdefault(ev, {}).setdefault(k, []).append((float(z[k][w].mean() - z[k][pre].mean()), float(z[k][pre].mean())))
    H[v][ev].setdefault('gates', []).append((z['gates'][pre].mean(0).tolist(), z['gates'][w].mean(0).tolist()))
N['head_dev'] = {v: {e: {k: dict(delta=round(float(np.mean([a[0] for a in x])), 4), pre=round(float(np.mean([a[1] for a in x])), 4)) for k, x in d.items() if k != 'gates'} for e, d in E_.items()} for v, E_ in H.items()}
print(json.dumps(N['head_dev'], indent=None)[:1800])
json.dump(N, open('results/analysis/numbers.json', 'w'), indent=1)
