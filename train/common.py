"""Shared training utilities: drone sets, episode pools, schedules, checkpoints,
NaN guards and JSONL metric logging."""
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sim import env as envm
from sim import sampler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = Path(os.environ.get('IGR_RESULTS', ROOT / 'results'))
LOGS = Path(os.environ.get('IGR_LOGS', ROOT / 'logs'))

_SETS = {}


def drone_set(name):
    if name not in _SETS:
        sampler.build_sets(DATA)
        _SETS[name] = {k: jnp.asarray(v) for k, v in sampler.load_set(DATA, name).items()}
    return _SETS[name]


def sample_pool(key, dset, spec, sev, n, p_truth=0.0, drone_idx=None, script=None):
    """n fresh episodes; `sev` may be a scalar or per-env array; `drone_idx` [n] fixes the drones
    (difficulty-weighted resampling, per-drone evaluation); `script` dict of [n] arrays (paired events)."""
    sev = jnp.broadcast_to(jnp.asarray(sev, jnp.float32), (n,))
    p_truth = jnp.broadcast_to(jnp.asarray(p_truth, jnp.float32), (n,))
    keys = jax.random.split(key, n)
    if drone_idx is None:
        drone_idx = jax.random.randint(jax.random.fold_in(key, 11), (n,), 0, dset['mass'].shape[0])
    if script is None:
        return jax.vmap(lambda k, s, p, i: envm.sample_episode(k, dset, spec, s, p, drone_idx=i))(
            keys, sev, p_truth, drone_idx)
    return jax.vmap(lambda k, s, p, i, sc: envm.sample_episode(k, dset, spec, s, p, script=sc, drone_idx=i))(
        keys, sev, p_truth, drone_idx, script)


def severity(frac, ramp=0.3):
    """Severity curriculum: 0 -> 1 linearly over the first 30 % of a stage."""
    return jnp.clip(frac / ramp, 0.0, 1.0)


def safe_norm(tree):
    """Scaled global L2 norm; report inf for non-finite inputs.

    Scaling avoids overflow from squaring large entries. A true norm larger than
    the dtype's maximum can still be inf; clipping must not divide by that norm.
    """
    leaves = [jnp.where(jnp.isfinite(x), x, 0.0).astype(jnp.float32) for x in jax.tree.leaves(tree)]
    if not leaves:
        return jnp.float32(0.0)
    m = jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in leaves]))
    s = jnp.where(m > 0, m, 1.0)
    norm = m * jnp.sqrt(sum(jnp.sum((x / s) ** 2) for x in leaves))
    return jnp.where(finite_tree(tree), norm, jnp.inf)


def stable_clip_by_global_norm(max_norm):
    """Clip in scaled coordinates, even when the raw norm cannot be represented.

    Non-finite entries are sanitized only to keep the candidate update safe;
    the caller must reject the entire update if its original gradient is invalid.
    """
    if max_norm <= 0:
        raise ValueError('max_norm must be positive')

    def update(updates, state, params=None):
        del params
        clean = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), updates)
        leaves = jax.tree.leaves(clean)
        if not leaves:
            return clean, state
        m = jnp.max(jnp.stack([jnp.max(jnp.abs(x).astype(jnp.float32)) for x in leaves]))
        denominator = jnp.where(m > 0, m, 1.0)
        unit = jax.tree.map(lambda x: x.astype(jnp.float32) / denominator, clean)
        unit_norm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(unit)))
        radius = max_norm / jnp.where(unit_norm > 0, unit_norm, 1.0)
        clipped = jax.tree.map(lambda x, u: jnp.where(m > radius, u * radius, x).astype(x.dtype), clean, unit)
        return clipped, state

    return optax.GradientTransformation(optax.init_empty_state, update)


def tree_where(mask, a, b):
    """Select per leading-axis entry (mask shape [N])."""
    def sel(x, y):
        m = mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))
        return jnp.where(m, x, y)
    return jax.tree.map(sel, a, b)


def finite_tree(tree):
    leaves = [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree) if jnp.issubdtype(x.dtype, jnp.floating)]
    return jnp.all(jnp.stack(leaves)) if leaves else jnp.asarray(True)


# --------------------------------------------------------------------------- io
def to_numpy(tree):
    return jax.tree.map(lambda x: np.asarray(x), tree)


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp{os.getpid()}')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=2, default=_json_default).encode())


def _json_default(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, 'tolist'):
        return np.asarray(x).tolist()
    return str(x)


class SelectionStore:
    """Selection checkpoints: separate from the resume checkpoints and ALL kept.
    Students every 100 epochs, teachers every 10 % of their budget."""

    def __init__(self, job_dir, sub='sel'):
        self.dir = Path(job_dir) / sub
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, tag, step, params, meta=None):
        atomic_write_bytes(self.dir / f'{tag}_{step:09d}.pkl', pickle.dumps(dict(params=to_numpy(params),
                                                                                 step=step, meta=meta or {})))

    def list(self, tag):
        return sorted(self.dir.glob(f'{tag}_*.pkl'))

    def steps(self, tag):
        return [int(f.stem.rsplit('_', 1)[1]) for f in self.list(tag)]

    def load(self, tag, step):
        d = pickle.loads((self.dir / f'{tag}_{step:09d}.pkl').read_bytes())
        return jax.tree.map(jnp.asarray, d['params'])


class Checkpointer:
    """Resume checkpoints (every `every_s` seconds and at stage end); keeps last 3."""

    def __init__(self, job_dir, every_s=600, keep=3):
        self.dir = Path(job_dir) / 'ckpt'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every_s = every_s
        self.keep = keep
        self.last = time.time()

    def due(self):
        return time.time() - self.last >= self.every_s

    def save(self, step, payload):
        payload = to_numpy(payload)
        atomic_write_bytes(self.dir / f'ckpt_{step:09d}.pkl', pickle.dumps(payload))
        for old in sorted(self.dir.glob('ckpt_*.pkl'))[:-self.keep]:
            old.unlink(missing_ok=True)
        self.last = time.time()

    def latest(self):
        files = sorted(self.dir.glob('ckpt_*.pkl'))
        for f in reversed(files):
            try:
                return pickle.loads(f.read_bytes())
            except Exception:  # corrupted partial file: fall back to an older one
                continue
        return None


def save_params(path, params, meta=None):
    atomic_write_bytes(path, pickle.dumps(dict(params=to_numpy(params), meta=meta or {})))


def load_params(path):
    d = pickle.loads(Path(path).read_bytes())
    return jax.tree.map(jnp.asarray, d['params']), d.get('meta', {})


class MetricLog:
    def __init__(self, job):
        self.path = LOGS / 'metrics' / f'{job}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **kw):
        kw['time'] = time.time()
        with open(self.path, 'a') as f:
            f.write(json.dumps(kw, default=_json_default) + '\n')


class NonFiniteError(RuntimeError):
    pass


class ThroughputMonitor:
    """Throughput ALERT. Baseline = median
    sps over the first 5 minutes of a stage; `add` returns True once when the rolling median has
    stayed < 60 % of it for 15 min, so the trainer can log it. Training is never altered."""

    def __init__(self, window_s=900.0, warmup_s=300.0, drop=0.4, enabled=True):
        self.t0 = time.time()
        self.window_s, self.warmup_s, self.drop, self.enabled = window_s, warmup_s, drop, enabled
        self.samples, self.baseline, self.low_since = [], None, None

    def add(self, sps):
        now = time.time()
        self.samples.append((now, sps))
        if not self.enabled:
            return False
        if self.baseline is None:
            if now - self.t0 >= self.warmup_s and len(self.samples) >= 5:
                self.baseline = float(np.median([s for _, s in self.samples[1:]]))
            return False
        recent = [s for t, s in self.samples if now - t <= 120]
        low = np.median(recent) < (1 - self.drop) * self.baseline
        self.low_since = (self.low_since or now) if low else None
        fire = bool(low and now - self.low_since >= self.window_s)
        if fire:
            self.low_since = None
        return fire


def lr_linear(base, frac):
    return base * max(0.0, 1.0 - frac)


def lr_cosine(base, final, frac):
    """Cosine base -> final over frac in [0, 1]; constant `final` afterwards (convergence extensions)."""
    return final + 0.5 * (base - final) * (1 + np.cos(np.pi * min(1.0, frac)))


def unsolved_mask(profile=None):
    """[n_teacher_seeds, n_train_drones] bool: train drones on which E_k succeeds < 80 %.
    The smoke profile computes the mask but does not apply it (its untrained E solves no drone, which
    would mask every label and leave the imitation losses untested)."""
    if profile is not None and not profile.get('apply_unsolved_mask', True):
        return None
    p = RESULTS / 'E' / 'unsolved.npy'
    return np.load(p) if p.exists() else None
