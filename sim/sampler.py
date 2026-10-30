"""RAPTOR-style ancestral drone sampler and fixed drone sets.

Drone sets are plain dicts of float32 arrays with a leading drone axis, saved
to data/drones_<name>.npz. Max torque uses per-rotor thrust (sqrt2 * l * f_max),
as in L2F, not total thrust.
"""
from pathlib import Path

import numpy as np

G = 9.81
C_F = np.array([0.032, 0.131, 0.837])  # normalized thrust curve, sums to 1
FIELDS = ('mass', 't2w', 't2i', 'l', 'cf', 'cm', 'T_up', 'T_down', 'J', 'cd',
          'sigma_fext', 'named')
SET_SEEDS = {'train': 1001, 'val': 2002, 'test': 3003, 'high_t2w': 4004}
SET_SIZES = {'train': 4096, 'val': 256, 'test': 512, 'high_t2w': 256}
# Drag coefficient scale, N s / m at l = 0.1 m  (U(0, 0.3); tuned).
CD_MAX_AT_L01 = 0.12


def _assemble(mass, t2w, t2i, l, cm, T_up, T_down, cd, sigma_fext, named,
              J=None):
    T_total = t2w * G * mass
    cf = C_F[None, :] * (T_total / 4.0)[:, None]
    tau_max = np.sqrt(2.0) * l * T_total / 4.0
    if J is None:
        jxx = tau_max / t2i
        J = np.stack([jxx, jxx, 1.832 * jxx], axis=1)
    else:
        t2i = tau_max / J[:, 0]
    d = dict(mass=mass, t2w=t2w, t2i=t2i, l=l, cf=cf, cm=cm, T_up=T_up,
             T_down=T_down, J=J, cd=cd, sigma_fext=sigma_fext, named=named)
    return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}


def sample_drones(n, seed, t2w_range=(1.5, 5.0)):
    rng = np.random.default_rng(seed)
    t2w = rng.uniform(*t2w_range, n)
    s = rng.uniform(np.cbrt(0.02), np.cbrt(5.0), n)
    mass = s ** 3
    r_ms = np.clip(rng.normal(7.24, 0.66, n), 5.5, 9.5)
    l = np.cbrt(mass) / r_ms
    t2i = rng.uniform(40, 1200, n)
    cm = rng.uniform(0.005, 0.05, n)
    T_up = rng.uniform(0.03, 0.1, n)
    T_down = rng.uniform(0.03, 0.3, n)
    sigma_fext = rng.uniform(0, 1, n) * 0.1 * (t2w - 1)
    cd = rng.uniform(0, CD_MAX_AT_L01, n) * (l / 0.1) ** 2
    return _assemble(mass, t2w, t2i, l, cm, T_up, T_down, cd, sigma_fext,
                     np.zeros(n))


# Named realistic platforms (approximate public figures).
NAMED = {
    #               mass   t2w   l      J (xx, yy, zz)               cm     Tup   Tdown  cd
    'crazyflie': (0.035, 1.75, 0.046, (1.66e-5, 1.66e-5, 2.93e-5), 0.005, 0.05, 0.15, 0.010),
    'fivein':    (0.60, 5.0, 0.110, (1.2e-3, 1.2e-3, 2.2e-3), 0.010, 0.03, 0.05, 0.10),
    'x500':      (1.20, 3.0, 0.250, (2.17e-2, 2.17e-2, 4.0e-2), 0.016, 0.06, 0.10, 0.30),
    'stretch_t2w10': (0.50, 10.0, 0.100, (7.0e-4, 7.0e-4, 1.28e-3), 0.010, 0.03, 0.06, 0.08),
}


def named_platforms():
    rows = list(NAMED.values())
    arr = lambda i: np.array([r[i] for r in rows], dtype=np.float64)
    J = np.array([r[3] for r in rows])
    return _assemble(arr(0), arr(1), None, arr(2), arr(4), arr(5), arr(6),
                     arr(7), np.full(len(rows), 0.05), np.arange(1, len(rows) + 1), J=J)


def concat_sets(a, b):
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in FIELDS}


def build_sets(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sets = {}
    for name in ('train', 'val', 'test', 'high_t2w'):
        t2w_range = (6.0, 12.0) if name == 'high_t2w' else (1.5, 5.0)
        d = sample_drones(SET_SIZES[name], SET_SEEDS[name], t2w_range)
        if name == 'test':
            d = concat_sets(d, named_platforms())
        sets[name] = d
    sets['named'] = named_platforms()
    for name, d in sets.items():
        path = data_dir / f'drones_{name}.npz'
        if not path.exists():
            np.savez(path, **d)
    return {name: load_set(data_dir, name) for name in sets}


def load_set(data_dir, name):
    with np.load(Path(data_dir) / f'drones_{name}.npz') as f:
        return {k: f[k].astype(np.float32) for k in FIELDS}


def n_drones(dset):
    return int(dset['mass'].shape[0])
