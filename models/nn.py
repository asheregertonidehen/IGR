"""Minimal pure-function NN primitives (param pytrees; vmappable over models).

GRU follows the RAPTOR / RLtools convention (gate order r, z, n; separate input
and hidden biases; h' = z * h + (1 - z) * n). minGRU follows Feng et al. 2024
(gates depend on the input only).
"""
import jax
import jax.numpy as jnp


def dense_init(key, n_in, n_out, scale=1.0):
    w = jax.nn.initializers.orthogonal(scale)(key, (n_in, n_out), jnp.float32)
    return dict(w=w, b=jnp.zeros(n_out))


def dense(p, x):
    return x @ p['w'] + p['b']


def mlp_init(key, sizes, out_scale=1.0):
    ks = jax.random.split(key, len(sizes) - 1)
    layers = [dense_init(k, a, b, jnp.sqrt(2.0)) for k, a, b in zip(ks[:-1], sizes[:-2], sizes[1:-1])]
    layers.append(dense_init(ks[-1], sizes[-2], sizes[-1], out_scale))
    return layers


def mlp(layers, x, act=jax.nn.elu):
    for p in layers[:-1]:
        x = act(dense(p, x))
    return dense(layers[-1], x)


def gru_init(key, n_in, n_h):
    k1, k2 = jax.random.split(key)
    wi = jax.nn.initializers.lecun_normal()(k1, (n_in, 3 * n_h))
    wh = jnp.concatenate([jax.nn.initializers.orthogonal()(k, (n_h, n_h)) for k in jax.random.split(k2, 3)], 1)
    return dict(wi=wi, wh=wh, bi=jnp.zeros(3 * n_h), bh=jnp.zeros(3 * n_h))


def gru(p, h, x):
    n_h = h.shape[-1]
    gi = x @ p['wi'] + p['bi']
    gh = h @ p['wh'] + p['bh']
    r = jax.nn.sigmoid(gi[..., :n_h] + gh[..., :n_h])
    z = jax.nn.sigmoid(gi[..., n_h:2 * n_h] + gh[..., n_h:2 * n_h])
    n = jnp.tanh(gi[..., 2 * n_h:] + r * gh[..., 2 * n_h:])
    return z * h + (1 - z) * n


def lstm_init(key, n_in, n_h):
    k1, k2 = jax.random.split(key)
    wi = jax.nn.initializers.lecun_normal()(k1, (n_in, 4 * n_h))
    wh = jnp.concatenate([jax.nn.initializers.orthogonal()(k, (n_h, n_h)) for k in jax.random.split(k2, 4)], 1)
    b = jnp.zeros(4 * n_h).at[n_h:2 * n_h].set(1.0)  # forget-gate bias 1
    return dict(wi=wi, wh=wh, b=b)


def lstm(p, hc, x):
    """hc = concat(h, c); returns new concat(h, c)."""
    n_h = hc.shape[-1] // 2
    h, c = hc[..., :n_h], hc[..., n_h:]
    g = x @ p['wi'] + h @ p['wh'] + p['b']
    i = jax.nn.sigmoid(g[..., :n_h])
    f = jax.nn.sigmoid(g[..., n_h:2 * n_h])
    o = jax.nn.sigmoid(g[..., 2 * n_h:3 * n_h])
    u = jnp.tanh(g[..., 3 * n_h:])
    c = f * c + i * u
    return jnp.concatenate([o * jnp.tanh(c), c], -1)


def mingru_init(key, n_in, n_h):
    k1, k2 = jax.random.split(key)
    return dict(z=dense_init(k1, n_in, n_h), h=dense_init(k2, n_in, n_h))


def mingru(p, h, x):
    z = jax.nn.sigmoid(dense(p['z'], x))
    return (1 - z) * h + z * dense(p['h'], x)


CELLS = {
    'gru': (gru_init, gru, 1),
    'lstm': (lstm_init, lstm, 2),     # state = (h, c)
    'mingru': (mingru_init, mingru, 1),
}


def cell_out(cell_type, state):
    """Hidden output h from a cell state (LSTM state is concat(h, c))."""
    return state[..., :state.shape[-1] // CELLS[cell_type][2]]


def count_params(tree):
    return int(sum(x.size for x in jax.tree.leaves(tree)))
