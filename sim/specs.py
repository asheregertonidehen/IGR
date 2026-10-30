"""Load configs/suites.yaml into jnp spec pytrees (one compiled program per shape)."""
from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SUITES_YAML = ROOT / 'configs' / 'suites.yaml'
META_KEYS = ('group', 'base', 'drones', 'family', 'event_at_peak')


@lru_cache(maxsize=None)
def raw():
    return yaml.safe_load(SUITES_YAML.read_text())


def to_jnp(d):
    return {k: jnp.asarray(np.asarray(v, dtype=np.float32)) for k, v in d.items() if k not in META_KEYS}


def train_spec():
    d = dict(raw()['train'])
    return to_jnp(d), {'drones': d['drones'], 'group': 'train'}


def _merge(*ds):
    out = {}
    for d in ds:
        out.update({k: v for k, v in d.items()})
    return out


def suite_dict(name, extra=None):
    r = raw()
    s = dict(r['suites'][name])
    base = s.get('base', 'nominal')
    layers = [r['train']]
    if base == 'nominal':
        layers.append(r['nominal'])
    layers += [r['eval_defaults'], s, extra or {}]
    d = _merge(*layers)
    d['group'] = s['group']
    return d


def suite_spec(name, extra=None):
    d = suite_dict(name, extra)
    return to_jnp(d), {'drones': d['drones'], 'group': d['group'], 'name': name}


def suite_names(groups=None):
    names = list(raw()['suites'])
    if groups is None:
        return names
    return [n for n in names if raw()['suites'][n]['group'] in groups]


def heldout_suites():
    return suite_names(('heldout',))


def family_of(suite):
    return raw()['suites'][suite].get('family')


def family_suites(fam):
    """Held-out suites of a family: 'S' sensing, 'A', 'D', 'K', 'C'."""
    return [s for s in heldout_suites() if family_of(s) == fam]


def families():
    return dict(raw()['families'])


def discrimination(protocol='primary'):
    """(jnp spec, settings, protocol dict) for one paired-event protocol."""
    r = raw()
    d = dict(r['discrimination'])
    prot = dict(d['protocols'][protocol])
    layers = [r['train']] + ([r['nominal']] if prot.get('base', 'nominal') == 'nominal' else [])
    spec = _merge(*layers, r['eval_defaults'], prot)
    keep = {k: v for k, v in d.items() if k != 'protocols'}
    return to_jnp(spec), keep, prot


def discrimination_protocols():
    return list(raw()['discrimination']['protocols'])
