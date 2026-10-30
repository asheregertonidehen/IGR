"""Innovation-gated representation cell, shared by the teacher, the student and the ablations.

    C_loc = {gyro, accel, attitude_est},  C_ext = {ext_pos, ext_vel, ref_stream}
    # 1) PREDICT from local sensing and the commands the vehicle knows it sent
    u        = [E_loc(x_loc), E_a(a_{t-6..t-1})]
    h_prior  = GRU_rep(h_prev, u)
    mu_c, logvar_c = P_c(h_prior)                  # channel expectation (standardized units)
    # 2) INNOVATION
    nu_c  = valid_c (x_c - mu_c);  nut_c = nu_c exp(-logvar_c / 2);  nis_c = clip(sum nut_c^2, 0, 50)
    # 3) MEMORY-WRITE GATES
    f_c   = E_nu(nut_c, |nut_c|, log1p(nis_c), valid_c)                    # 2-layer ELU MLP
    g_c   = sigmoid(w_c . [h_prior, f_c] + b_c),  b_c init +2
    # 4) CORRECT; g is appended so "ignored" differs from "observed zero"
    h = tanh(h_prior + kappa tanh(W [h_prior, g_c f_c, g_c]));  r = Linear(h)
    # 5) CONTROL: representation with gated raw data, local sensing and history
    z = [r, g_c E_c(x_c), g_c, E_loc(x_loc), E_a(a)]  (+ teacher only: drone_emb, m E_truth(x_truth), m)
    h_ctl = GRU_ctl(h_ctl_prev, z);  a = tanh(Linear([h_ctl, r]))

IGR-sup (`Full`) supervises P_c on the true channel value and reads it through a stop-gradient.
IGR-task (`A3c`) sets lam_pc = 0, so control gradients train P_c.

Config switches are float scalars so variants vmap in one program: use_gates (0: g = 1, A2),
use_innov (0: raw embeddings replace f_c, A5), rep_grad (0: control gradients stopped from every
representation parameter, A4), corr_scale (kappa).
"""
import jax
import jax.numpy as jnp

from models import heads as hd
from models import inputs as I
from models import nn

GATE_BIAS = 2.0
LOGVAR_CLIP = (-8.0, 4.0)
NIS_CLIP = 50.0
REP_MODULE = ('rep', 'pc', 'enu', 'gate', 'corr', 'r', 'heads')


def widths(width=1):
    """Student cell (width 1): GRU_rep 24, GRU_ctl 16, r 16. Teacher (width 3): 72 / 48 / 32 with
    channel embeddings and E_nu doubled."""
    m = 2 if width > 1 else 1
    return dict(h_rep=24 * width, h_ctl=16 * width, r=16 if width == 1 else 32,
                loc=dict(gyro=4 * m, accel=4 * m, att=4 * m), act=8 * m, f=8 * m, enu_h=12 * m,
                drone=16, truth=8 * m)


class GatedCell:
    kind = 'gated'

    def __init__(self, cell_type='gru', teacher=False, width=1):
        self.cell_type, self.teacher, self.width = cell_type, teacher, width
        self.W = widths(width)
        self.k = nn.CELLS[cell_type][2]
        W = self.W
        self.el = sum(W['loc'].values())
        self.lead = (W['drone'] + W['truth'] + 1) if teacher else 0      # teacher-only input slots
        self.u_dim = self.el + W['act'] + self.lead
        self.z_dim = W['r'] + 3 * W['f'] + 3 + self.el + W['act'] + self.lead
        self.ext_slices, off = {}, 0
        for c in I.EXT_CHANNELS:
            self.ext_slices[c] = slice(off, off + I.EXT_VALUE_DIMS[c])
            off += I.EXT_VALUE_DIMS[c]

    # ------------------------------------------------------------------ params
    def init(self, key):
        W = self.W
        ks = iter(jax.random.split(key, 64))
        cinit = nn.CELLS[self.cell_type][0]
        p = dict(loc={c: nn.dense_init(next(ks), I.CH_DIMS[c], W['loc'][c]) for c in I.LOC_CHANNELS},
                 act=nn.dense_init(next(ks), I.HIST_DIM, W['act']),
                 rep=cinit(next(ks), self.u_dim, W['h_rep']),
                 pc=nn.dense_init(next(ks), W['h_rep'], 2 * I.PC_DIM, 0.1),
                 enu={c: nn.mlp_init(next(ks), [2 * I.EXT_VALUE_DIMS[c] + 3, W['enu_h'], W['f']])
                      for c in I.EXT_CHANNELS},
                 gate={c: dict(nn.dense_init(next(ks), W['h_rep'] + W['f'], 1, 0.1), b=jnp.full(1, GATE_BIAS))
                       for c in I.EXT_CHANNELS},
                 corr=nn.dense_init(next(ks), W['h_rep'] + 3 * W['f'] + 3, W['h_rep'], 0.5),
                 r=nn.dense_init(next(ks), W['h_rep'], W['r']),
                 raw={c: nn.dense_init(next(ks), I.CH_DIMS[c], W['f']) for c in I.EXT_CHANNELS},
                 ctl=cinit(next(ks), self.z_dim, W['h_ctl']),
                 out=nn.dense_init(next(ks), W['h_ctl'] + W['r'], 4, 0.01),
                 heads=hd.init_heads(next(ks), W['h_rep']),
                 log_std=jnp.full(4, -0.5))
        if self.teacher:
            p['drone'] = nn.mlp_init(next(ks), [I.DESC_DIM, 32, W['drone']])
            p['truth'] = nn.dense_init(next(ks), I.TRUTH_DIM, W['truth'])
        return p

    def deploy_keys(self, cfg):
        """Parameter groups a variant uses at deployment (heads, log_std are training-only)."""
        keys = {'loc', 'act', 'rep', 'r', 'raw', 'ctl', 'out', 'corr'}
        if float(cfg['use_gates']) > 0:
            keys |= {'gate'}
        if float(cfg['use_innov']) > 0:
            keys |= {'pc', 'enu'}
        if self.teacher:
            keys |= {'drone', 'truth'}
        return keys

    def deploy_param_count(self, p, cfg):
        keys = self.deploy_keys(cfg)
        return nn.count_params({k: v for k, v in p.items() if k in keys})

    def init_carry(self):
        W = self.W
        return dict(rep=jnp.zeros(W['h_rep'] * self.k), ctl=jnp.zeros(W['h_ctl'] * self.k), r=jnp.zeros(W['r']))

    # ------------------------------------------------------------------ inputs
    def features(self, o, pv, cfg):
        ch = I.channels(o)
        if self.teacher:
            ch['truth'] = I.truth_channel(pv)
            ch['descriptor'] = I.descriptor(pv)
        return ch

    # ------------------------------------------------------------------ step
    def step(self, p, cfg, carry, ch):
        cell = nn.CELLS[self.cell_type][1]
        W = self.W
        h_ctl_prev = nn.cell_out(self.cell_type, carry['ctl'])
        rg = cfg['rep_grad']
        mix = lambda x: rg * x + (1 - rg) * jax.lax.stop_gradient(x)
        use_g, use_innov = cfg['use_gates'], cfg['use_innov']

        e_loc = jnp.concatenate([nn.dense(p['loc'][c], ch[c]) for c in I.LOC_CHANNELS])
        e_a = nn.dense(p['act'], ch['prev_actions'])
        e_raw = {c: nn.dense(p['raw'][c], ch[c]) for c in I.EXT_CHANNELS}
        lead = []
        if self.teacher:
            m = ch['truth'][-1:]
            lead = [nn.mlp(p['drone'], ch['descriptor']), m * nn.dense(p['truth'], ch['truth']), m]
        # 1) predict; privileged teacher inputs reach the controller only (zero slots keep the input size)
        u = [e_loc, e_a] + [0.0 * x for x in lead]
        rep_state = cell(p['rep'], carry['rep'], jnp.concatenate(u))
        h_prior = nn.cell_out(self.cell_type, rep_state)
        pc = nn.dense(p['pc'], h_prior)
        mu, logvar = pc[:I.PC_DIM], jnp.clip(pc[I.PC_DIM:], *LOGVAR_CLIP)
        # When P_c is supervised (lam_pc > 0) the innovation reads it through a stop-gradient: forward values are
        # unchanged and the backward loop h -> mu -> nu / sigma -> delta -> h is cut (its gain reaches 1/sigma).
        ground = (cfg['lam_pc'] > 0).astype(jnp.float32)
        sg = jax.lax.stop_gradient
        mu_i = ground * sg(mu) + (1 - ground) * mu
        logvar_i = ground * sg(logvar) + (1 - ground) * logvar
        x_std = I.standardize(I.ext_values(ch), 'pc')
        valid = I.ext_valid(ch)
        corr_in, gslots, ctl_raw, nis_all = [], [], [], []
        for i, c in enumerate(I.EXT_CHANNELS):
            sl = self.ext_slices[c]
            v = valid[i]
            # 2) innovation, normalized by the predicted scale
            nu = v * (x_std[sl] - mu_i[sl])
            nut = nu * jnp.exp(-0.5 * logvar_i[sl])
            nis = jnp.clip(jnp.sum(nut * nut), 0.0, NIS_CLIP)
            f = nn.mlp(p['enu'][c], jnp.concatenate([nut, jnp.abs(nut), jnp.log1p(nis)[None], v[None], ch[c][-1:]]))
            feat = use_innov * f + (1 - use_innov) * e_raw[c]
            # 3) memory-write gate
            g = jax.nn.sigmoid(nn.dense(p['gate'][c], jnp.concatenate([h_prior, feat])))[0]
            g = use_g * g + (1 - use_g)
            corr_in.append(g * feat)
            gslots.append(g)
            ctl_raw.append(mix(g) * e_raw[c])
            nis_all.append(nis)
        # 4) correct
        gs = jnp.stack(gslots)
        # corr_scale (kappa) damps the recurrent correction path; the outer tanh keeps memory in the GRU's range
        delta = cfg['corr_scale'] * jnp.tanh(nn.dense(p['corr'], jnp.concatenate([h_prior] + corr_in + [gs])))
        h_rep = jnp.tanh(h_prior + delta)
        rep_state = jnp.concatenate([h_rep, rep_state[W['h_rep']:]]) if self.k > 1 else h_rep
        r = nn.dense(p['r'], h_rep)
        r_c = mix(r)
        if 'r_a' in ch:   # reliance test: r seen by the controller = a r + b
            r_c = ch['r_a'] * r_c + ch['r_b']
        # 5) control
        z = jnp.concatenate([r_c, jnp.concatenate(ctl_raw), mix(gs), e_loc, e_a] + lead)
        ctl_state = cell(p['ctl'], carry['ctl'], z)
        h_ctl = nn.cell_out(self.cell_type, ctl_state)
        mean = nn.dense(p['out'], jnp.concatenate([h_ctl, r_c]))
        new = dict(rep=rep_state, ctl=ctl_state, r=r)
        aux = dict(gates=gs, nis=jnp.stack(nis_all), h_rep=h_rep, pc_mu=mu, pc_logvar=logvar,
                   r=r, hidden=jnp.concatenate([rep_state, ctl_state, r]),
                   truth_head=hd.truth_head(p['heads'], h_rep), id_head=hd.id_head(p['heads'], h_rep))
        return new, mean, aux
