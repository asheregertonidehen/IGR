"""Stage S: student distillation, RAPTOR's on-policy imitation extended.

for epoch in 1..N:
    N_env episodes of 500 steps per model, driven by the labeling teacher for the first 10 epochs
    (RAPTOR EPOCH_TEACHER_FORCING = 10), else by the student (on-policy);
    label = a_T (T_k) or a_E (E_k) per variant, masked on E_k's unsolved drones;
    loss = MSE(a_S, label) + lambda_s * (sum_k lambda_k event-weighted L_k + lambda_T cap + lambda_g gates);
    one shuffled pass in minibatches of 64 full 500-step sequences (RAPTOR BATCH_SIZE), rematerialized
    BPTT over the family's backward window (family_bptt); Adam lr from configs/run.yaml (constant 1e-4), grad clip 1.0; per-model optimizer, lr and NaN guard.

A model on teacher seed k uses E_k and T_k for every label (actions, heads, P_c, gates).
Environments depend only on the (teacher seed, student seed) pair, so paired variants see the same
training episodes until their on-policy behaviour diverges; models of a pair share their init.
A family job runs its models as one or more sequential vmapped programs (split by the static
estimator mode; per-model computation is identical either way).
Selection checkpoints every 100 epochs (all kept) feed the select_ckpt job (best val_id checkpoint).
"""
import json
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from eval import modelsets
from models import heads as hd
from models import inputs as I
from models import variants as V
from models.expert import Expert
from sim import env as envm
from sim import specs
from train import common, diagnostics, losses

EXPERT = Expert()
COPYCAT_EPOCHS = (1, 10, 11, 15, 30)
LR_INIT = 1e-3   # only the optimizer's initial hyperparameter; the learning rate of every update is set from the config (lr_at)
REQUIRED = ('lr', 'lr_final', 'bptt')   # no silent defaults: a missing key must stop the job


def lr_at(b, frac):
    """Learning rate at progress `frac` of the base budget: cosine from b['lr'] to b['lr_final']
    (constant if they are equal; constant lr_final beyond the base budget)."""
    return common.lr_cosine(float(b['lr']), float(b['lr_final']), frac)


def check_settings(b):
    missing = [k for k in REQUIRED if k not in b]
    if missing:
        raise RuntimeError(f'student budget lacks {missing}: refusing to fall back to defaults (configs/run.yaml)')
    return dict(bptt=int(b['bptt']) if b['bptt'] else None, lr=float(b['lr']), lr_final=float(b['lr_final']))


def family_bptt(b, fam):
    """Backward window of a family: reused baselines keep their trained horizon, others use b['bptt']."""
    return int((b.get('bptt_by_family') or {}).get(fam, b['bptt']))


def _opt():
    return optax.chain(common.stable_clip_by_global_norm(1.0),
                       optax.inject_hyperparams(optax.adam)(learning_rate=LR_INIT))


def convergence_state():
    p = common.RESULTS / 'convergence.json'
    return json.loads(p.read_text()) if p.exists() else dict(n_ext=0, history=[])


def target_epochs(b):
    return int(round(int(b['epochs']) * 1.5 ** convergence_state()['n_ext']))


class Distiller:
    def __init__(self, fam, models, n_envs, t_params, t_names, e_params, e_seeds, unsolved=None, ep_steps=500,
                 mb_seqs=64, passes=1, est=None, lam_star=None, tuning=None, bptt=None):
        self.fam, self.models = fam, models
        self.M, self.N, self.T = len(models), n_envs, ep_steps
        self.arch = modelsets.fam_arch(fam)
        self.has_heads = self.arch.kind == 'gated' or getattr(self.arch, 'heads', False)
        self.gated = self.arch.kind == 'gated'
        self.tprog = V.FAMILIES[fam]['teachers']
        self.tarch = modelsets.teacher_arch(self.tprog)
        self.cfg = V.stack_cfg([m[1] for m in models], lam_star=lam_star, est_tuning=tuning)
        lab_T = [V.VARIANTS[m[1]]['label'] > 0 for m in models]
        self.need_T = any(lab_T) or (self.has_heads and any(float(x) > 0 for x in self.cfg['lam_T']))
        self.need_cap = self.has_heads and self.need_T
        self.need_E = not all(lab_T)
        self.t_idx = jnp.asarray([t_names.index(m[4]) for m in models], jnp.int32)
        self.e_idx = jnp.asarray([e_seeds.index(m[2]) for m in models], jnp.int32)
        self.pair = jnp.asarray([100 * m[2] + m[3] for m in models], jnp.int32)
        self.t_params, self.e_params = t_params, e_params
        self.tcfg = {k: jnp.stack([V.teacher_cfg(m[4])[k] for m in models]) for k in V.CFG_KEYS}
        n_dr = int(common.drone_set('train')['mass'].shape[0])
        self.lab_mask = jnp.asarray(np.stack([~unsolved[int(i)] for i in self.e_idx]) if unsolved is not None
                                    else np.ones((self.M, n_dr), bool))
        self.est = est
        self.spec, _ = specs.train_spec()
        self.dset = common.drone_set('train')
        assert n_envs % mb_seqs == 0, (n_envs, mb_seqs)
        self.n_mb, self.passes = n_envs // mb_seqs, passes
        if bptt and bptt < ep_steps and ep_steps % bptt:
            raise ValueError(f'bptt {bptt} must divide the episode length {ep_steps}')
        self.bptt = bptt if bptt and bptt < ep_steps else None
        self.opt = _opt()
        self.rollout = jax.jit(self._rollout)
        self.update = jax.jit(self._update)
        self.diag = jax.jit(self._diag)

    def init_params(self, key):
        # Same initialization for every variant sharing a (teacher seed, student seed) pair.
        keys = jax.vmap(lambda s: jax.random.fold_in(key, s))(self.pair)
        params = jax.vmap(self.arch.init)(keys)
        return params, jax.vmap(self.opt.init)(params)

    # ----------------------------------------------------------------- rollout
    def _pools(self, key, sev):
        sev_m = sev * self.cfg['sev_mult']
        return jax.vmap(lambda s, sv: common.sample_pool(jax.random.fold_in(key, s), self.dset, self.spec, sv, self.N)
                        )(self.pair, sev_m)

    def _reset(self, pools):
        env = jax.vmap(jax.vmap(lambda e: envm.reset(e, self.est)))(pools)
        return jax.vmap(lambda e, c: jax.vmap(lambda ee: envm.with_est(ee, V.est_params(c)))(e))(env, self.cfg)

    def _rollout(self, params, key, sev, warm):
        k1, k2, k3 = jax.random.split(key, 3)
        env = self._reset(self._pools(k1, sev))
        fresh = self._reset(self._pools(k2, sev))
        bc = lambda c: jax.tree.map(lambda x: jnp.broadcast_to(x, (self.M, self.N) + x.shape), c)
        c_s0, c_t0 = bc(self.arch.init_carry()), bc(self.tarch.init_carry())
        tp = jax.tree.map(lambda x: x[self.t_idx], self.t_params) if self.need_T else None
        ep = jax.tree.map(lambda x: x[self.e_idx], self.e_params) if self.need_E else None
        vmn = lambda f: jax.vmap(jax.vmap(f, (None, None, 0, 0, 0)), (0, 0, 0, 0, 0))
        estep = jax.vmap(jax.vmap(lambda s, a, k: envm.step(s, a, k, self.est)))

        def student(p, cfg, c, o, pv):
            ch = self.arch.features(o, pv, cfg)
            c, mean, _ = self.arch.step(p, cfg, c, ch)
            return c, jnp.tanh(mean), ch

        def teacher(p, cfg, c, o, pv):
            ch = self.tarch.features(o, pv, cfg)
            c, mean, aux = self.tarch.step(p, cfg, c, ch)
            return c, jnp.tanh(mean), (aux['h_rep'], aux['pc_mu'], aux['gates'])

        def one(state, k):
            env, cs, ct = state
            o = env['obs']
            pv = jax.vmap(jax.vmap(envm.privileged))(env)
            cs, a_s, ch = vmn(student)(params, self.cfg, cs, o, pv)
            rec = dict(ch=ch)
            label = jnp.zeros_like(a_s)
            if self.need_T:
                ct, a_t, (h_t, pc_t, g_t) = vmn(teacher)(tp, self.tcfg, ct, o, pv)
                label = a_t
                if self.need_cap:
                    rec.update(h_t=h_t, pc_t=pc_t, g_t=g_t)
            if self.need_E:
                x_e = jax.vmap(jax.vmap(I.expert))(o, pv)
                a_e = jnp.tanh(jax.vmap(lambda p_, x: EXPERT.mean(p_, x))(ep, x_e))
                label = jnp.where(self.cfg['label'][:, None, None] > 0, label, a_e)
            a = jnp.where(warm, label, a_s)
            lab_w = jax.vmap(lambda msk, d: msk[d])(self.lab_mask, env['ep']['drone_idx']).astype(jnp.float32)
            keys = jax.vmap(lambda s: jax.random.split(jax.random.fold_in(k, s), self.N))(self.pair)
            env2, r, term = estep(env, a, keys)
            rec.update(lab=label, a=a, done=term, lab_w=lab_w, r=r,
                       err_s=jnp.sum(jnp.mean((a_s - label) ** 2, -1) * lab_w, -1) / jnp.maximum(jnp.sum(lab_w, -1), 1))
            if self.has_heads:
                v2 = lambda f: jax.vmap(jax.vmap(f))
                rec.update(truth=v2(I.truth_targets)(pv), idt=v2(I.id_targets)(pv), sig=v2(I.pred_signal)(o),
                           ev=pv['event_active'])
                if self.gated:
                    rec['pc'] = v2(I.pc_targets)(pv)
            sel = lambda new, old: jax.vmap(lambda d, n_, o_: common.tree_where(d, n_, o_))(term, new, old)
            ct_new = sel(c_t0, ct) if self.need_T else ct
            return (sel(fresh, env2), sel(c_s0, cs), ct_new), rec

        _, data = jax.lax.scan(one, (env, c_s0, c_t0), jax.random.split(k3, self.T))
        data = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), data)       # [T, M, ...] -> [M, T, ...]
        stats = dict(reward=data['r'].mean((1, 2)), term_rate=data['done'].mean((1, 2)),
                     online_mse=data.pop('err_s').mean(1), lab_frac=data['lab_w'].mean((1, 2)))
        data.pop('r')
        return data, stats

    # ----------------------------------------------------------------- losses (one model)
    def _forward(self, p, cfg, d):
        arch = self.arch
        n = d['done'].shape[1]
        init = jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), arch.init_carry())
        done_prev = jnp.concatenate([jnp.zeros_like(d['done'][:1]), d['done'][:-1]], 0)
        keep = ('h_rep', 'truth_head', 'id_head', 'pc_mu', 'pc_logvar', 'gates') if self.has_heads else ()

        @jax.checkpoint
        def step(carry, x):
            ch, dp = x
            carry = common.tree_where(dp, init, carry)
            carry, mean, aux = jax.vmap(lambda c, z: arch.step(p, cfg, c, z))(carry, ch)
            return carry, (mean, {k: aux[k] for k in keep if k in aux})
        if self.bptt is None:
            _, (mean, outs) = jax.lax.scan(step, init, (d['ch'], done_prev))
            return mean, outs
        # truncated BPTT windows with carried (gradient-stopped) state
        nw = self.T // self.bptt
        resh = lambda x: x.reshape((nw, self.bptt) + x.shape[1:])
        xs = jax.tree.map(resh, (d['ch'], done_prev))

        def window(carry, x):
            carry, ys = jax.lax.scan(step, jax.lax.stop_gradient(carry), x)
            return carry, ys
        _, (mean, outs) = jax.lax.scan(window, init, xs)
        flat = lambda x: x.reshape((self.T,) + x.shape[2:])
        return flat(mean), jax.tree.map(flat, outs)

    def _losses(self, p, cfg, th, d):
        mean, outs = self._forward(p, cfg, d)
        lw = d['lab_w']
        l_act = jnp.sum(jnp.mean((jnp.tanh(mean) - d['lab']) ** 2, -1) * lw) / jnp.maximum(jnp.sum(lw), 1.0)
        parts = dict(act=l_act)
        if not self.has_heads:
            return parts, outs
        w = losses.event_weights(d['ev'], cfg['beta'])
        parts.update(losses.rep_losses(cfg, p['heads'], outs, d, w))
        if self.need_cap:
            w_cap = losses.event_weights(d['ev'], cfg['beta'], mask=lw)
            outs_t = dict(h_rep=d['h_t'], truth_head=hd.truth_head(th, d['h_t']), id_head=hd.id_head(th, d['h_t']),
                          pc_mu=d['pc_t'])
            parts['cap'], comp = losses.capability_losses(cfg, p['heads'], outs, th, outs_t, d['a'], d['done'], w_cap)
            parts.update({f'cap_{k}': v for k, v in comp.items()})
            if 'gates' in outs:
                parts['gate'] = losses.mse_loss(outs['gates'], d['g_t'], w_cap)
        return parts, outs

    def _total(self, cfg, parts):
        if not self.has_heads:
            return parts['act']
        rep = (cfg['lam_pred'] * parts['pred'] + cfg['lam_truth'] * parts['truth'] + cfg['lam_id'] * parts['id']
               + cfg['lam_pc'] * parts.get('pc', 0.0) + cfg['lam_T'] * parts.get('cap', 0.0)
               + cfg['lam_g'] * parts.get('gate', 0.0))
        return parts['act'] + cfg['lam_scale'] * rep

    def _loss_one(self, p, cfg, th, d):
        parts, outs = self._losses(p, cfg, th, d)
        total = self._total(cfg, parts)
        st = dict(parts, loss=total)
        if 'gates' in outs:
            st['gate_mean'] = jnp.mean(outs['gates'])
        return total, st

    def _t_heads(self):
        return jax.tree.map(lambda x: x[self.t_idx], self.t_params['heads']) if self.need_cap else jnp.zeros(self.M)

    def _update(self, params, opt_state, data, key, lr):
        """data model-major [M, T, N, ...]; lr [M]. Returns params, opt_state, stats [M], finite [M]."""
        mb = self.N // self.n_mb

        def model_update(p, o, cfg, th, d, k, lr_m):
            o[1].hyperparams['learning_rate'] = lr_m

            def mb_step(c, idx):
                p, o = c
                dd = jax.tree.map(lambda x: x[:, idx], d)
                (loss, st), g = jax.value_and_grad(self._loss_one, has_aux=True)(p, cfg, th, dd)
                g_ok = common.finite_tree(g)
                gn = common.safe_norm(g)
                g = jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), g)
                upd, o2 = self.opt.update(g, o, p)
                p2 = optax.apply_updates(p, upd)
                # A large finite gradient is clipped, not discarded by an arbitrary
                # raw-norm threshold. Reject invalid loss/gradient/update transactions.
                loss_ok = jnp.isfinite(loss)
                update_ok = common.finite_tree(p2) & common.finite_tree(o2)
                accept = loss_ok & g_ok & update_ok
                p2 = jax.tree.map(lambda a, b: jnp.where(accept, a, b), p2, p)
                o2 = jax.tree.map(lambda a, b: jnp.where(accept, a, b), o2, o)
                st['grad_norm'] = gn
                st['grad_rejected'] = 1.0 - accept.astype(jnp.float32)
                st['loss_nonfinite'] = (~loss_ok).astype(jnp.float32)
                st['grad_nonfinite'] = (~g_ok).astype(jnp.float32)
                st['update_nonfinite'] = (~update_ok).astype(jnp.float32)
                return (p2, o2), st
            perms = jnp.concatenate([jax.random.permutation(jax.random.fold_in(k, i), self.N).reshape(self.n_mb, mb)
                                     for i in range(self.passes)])
            (p2, o2), st = jax.lax.scan(mb_step, (p, o), perms)
            peak_norm = jnp.max(st['grad_norm'])
            st = jax.tree.map(jnp.mean, st)
            st['grad_norm_max'] = peak_norm
            # A rejected minibatch must not roll back accepted minibatches. If all
            # updates were rejected, report failure rather than silent progress.
            ok = common.finite_tree(p2) & common.finite_tree(o2) & (st['grad_rejected'] < 1.0)
            p2 = jax.tree.map(lambda a, b: jnp.where(ok, a, b), p2, p)
            o2 = jax.tree.map(lambda a, b: jnp.where(ok, a, b), o2, o)
            return p2, o2, st, ok
        keys = jax.random.split(key, self.M)
        return jax.vmap(model_update)(params, opt_state, self.cfg, self._t_heads(), data, keys, lr)

    def _diag(self, params, data):
        """Gradient cosine, head R^2 (clean vs event steps) and loss magnitudes on a held-in batch."""
        n = min(16, self.N)

        def one(p, cfg, th, d):
            d = jax.tree.map(lambda x: x[:, :n], d)
            part = lambda name: (lambda q: self._losses(q, cfg, th, d)[0][name])
            names = [k for k in ('pred', 'truth', 'id', 'pc') if k in self._losses(p, cfg, th, d)[0]]
            out = diagnostics.cosines(part('act'), {k: part(k) for k in names}, p)
            parts, outs = self._losses(p, cfg, th, d)
            out.update(diagnostics.head_r2(outs, d))
            lam = dict(pred=cfg['lam_pred'], truth=cfg['lam_truth'], id=cfg['lam_id'], pc=cfg['lam_pc'],
                       cap=cfg['lam_T'], gate=cfg['lam_g'])
            out.update({f'rel_{k}': cfg['lam_scale'] * lam[k] * v / jnp.maximum(parts['act'], 1e-8)
                        for k, v in parts.items() if k in lam})
            return out
        return jax.vmap(one)(params, self.cfg, self._t_heads(), data)


# ----------------------------------------------------------------------------- job
def plan_programs(fam, models, max_models):
    """Split a family into sequential programs: by static estimator mode, then by memory."""
    progs = []
    for est, idx in modelsets.by_est(models).items():
        step = max(1, int(max_models or len(idx)))
        for s in range(0, len(idx), step):
            progs.append((est, idx[s:s + step]))
    return progs


def _load_teachers(fam):
    prog = V.FAMILIES[fam]['teachers']
    t_params, t_meta = common.load_params(common.RESULTS / prog / 'params.pkl')
    e_params, e_meta = common.load_params(common.RESULTS / 'E' / 'params.pkl')
    return t_params, [m[0] for m in t_meta['models']], t_meta, e_params, [m[2] for m in e_meta['models']]


def run(ctx, fam):
    b, prof = ctx.budget, ctx.profile
    I.target_stats(required=True)
    models = V.family_models(fam, prof)
    stage = V.FAMILIES[fam]['stage']
    lam = modelsets.lambda_star()
    if stage == 'B' and lam is None:
        raise RuntimeError(f'{fam}: lambda_s* not selected yet (select_lambda must run first)')
    tuning = modelsets.est_tuning()
    if any(m[1] in V.EST_TUNED for m in models) and tuning is None:
        raise RuntimeError(f'{fam}: classical estimator tuning missing (tune_classical must run first)')
    t_params, t_names, t_meta, e_params, e_seeds = _load_teachers(fam)
    unsolved = common.unsolved_mask(prof)
    n_epochs = target_epochs(b)
    base_epochs = int(b['epochs'])
    settings = dict(check_settings(b), bptt=family_bptt(b, fam))
    ctx.log(f"{fam}: EFFECTIVE student training settings: bptt={settings['bptt']} (state carried across windows), "
            f"lr {settings['lr']:g} -> {settings['lr_final']:g}, clip=stable_clip_by_global_norm(1.0), "
            f"no raw-norm rejection, n_envs={b['n_envs']}, epochs={b['epochs']}, minibatch_seqs={b.get('minibatch_seqs', 64)}")
    progs = plan_programs(fam, models, (b.get('max_models') or {}).get(fam))
    sel_store = common.SelectionStore(ctx.job_dir)
    metas = []
    for pi, (est, idx) in enumerate(progs):
        pm = [models[i] for i in idx]
        tag = f'{fam}_p{pi}'
        dist = Distiller(fam, pm, b['n_envs'], t_params, t_names, e_params, e_seeds, unsolved,
                         ep_steps=b.get('ep_steps', 500), mb_seqs=b.get('minibatch_seqs', 64), passes=b.get('passes', 1),
                         est=est, lam_star=lam, tuning=tuning, bptt=settings['bptt'])
        assert dist.bptt == (settings['bptt'] if settings['bptt'] and settings['bptt'] < dist.T else None)
        ctx.log(f'{tag}: {len(pm)} models (est={est}), {n_epochs} epochs x {dist.N} envs x {dist.T} steps, '
                f'{dist.n_mb * dist.passes} Adam steps/epoch')
        metas.append(train_program(ctx, dist, tag, n_epochs, base_epochs, b, sel_store, jax.random.PRNGKey(1000)))
    # family-level final params (selection happens in select_ckpt_<fam>)
    order = [i for _, idx in progs for i in idx]
    finals = [common.load_params(ctx.job_dir / f'final_{fam}_p{pi}.pkl')[0] for pi in range(len(progs))]
    stacked = jax.tree.map(lambda *x: jnp.concatenate(x), *finals)
    models_o = [models[i] for i in order]
    names = [m[0] for m in models_o]
    deploy = {m[0]: V.deploy_params(fam, m[1]) for m in models_o}
    meta = dict(models=[list(m) for m in models_o], programs=[dict(tag=f'{fam}_p{pi}', est=est, models=[models[i][0] for i in idx])
                                                            for pi, (est, idx) in enumerate(progs)],
                epochs=n_epochs, base_epochs=base_epochs, lambda_star=lam, est_tuning=tuning, settings=settings,
                failed=sum((m['failed'] for m in metas), []), copycat={k: v for m in metas for k, v in m['copycat'].items()},
                deploy_params=deploy, teachers={m[0]: m[4] for m in models_o},
                teacher_paths={n: t_meta['teachers'][n]['path'] for n in t_names},
                env_steps_per_model=n_epochs * b['n_envs'] * b.get('ep_steps', 500), names=names)
    common.save_params(ctx.job_dir / 'params.pkl', stacked, meta)
    common.atomic_write_json(ctx.job_dir / 'meta.json', meta)
    return dict(epochs=n_epochs, programs=len(progs), failed=meta['failed'])


def train_program(ctx, dist, tag, n_epochs, base_epochs, b, sel_store, key):
    names = [m[0] for m in dist.models]
    final_path = ctx.job_dir / f'final_{tag}.pkl'
    params, opt_state = dist.init_params(key)
    epoch, lr_mult, nan_counts, copycat = 0, np.ones(dist.M), np.zeros(dist.M, int), {}
    if final_path.exists():
        d = pickle.loads(final_path.read_bytes())
        if d['meta']['epoch'] >= n_epochs:
            ctx.log(f'{tag}: already trained to epoch {d["meta"]["epoch"]}')
            return d['meta']
        st = d['meta']['state']
        params, opt_state = jax.tree.map(jnp.asarray, d['params']), jax.tree.map(jnp.asarray, st['opt_state'])
        epoch, lr_mult, nan_counts = d['meta']['epoch'], np.asarray(st['lr_mult']), np.asarray(st['nan_counts'])
        copycat = d['meta'].get('copycat', {})
        ctx.log(f'{tag}: extending from epoch {epoch} to {n_epochs} (convergence extension, lr {float(b['lr_final']):g})')
    ck = ctx.ckpt.latest()
    if ck is not None and ck.get('tag') == tag and ck['epoch'] > epoch:
        params, opt_state = jax.tree.map(jnp.asarray, ck['params']), jax.tree.map(jnp.asarray, ck['opt_state'])
        epoch, lr_mult, nan_counts = ck['epoch'], np.asarray(ck['lr_mult']), np.asarray(ck['nan_counts'])
        copycat = ck.get('copycat', {})
        ctx.log(f'resumed {tag} at epoch {epoch}')
    failed = nan_counts > 3
    n_warm = int(b.get('warmup_epochs', 10))
    sel_every = int(b.get('selection_ckpt_every_epochs', 100))
    diag_every = max(1, int(np.ceil(diagnostics.DIAG_EVERY_UPDATES / (dist.n_mb * dist.passes))))
    t_last, alerted = time.time(), False
    while epoch < n_epochs:
        frac = epoch / base_epochs
        sev = float(common.severity(frac))
        warm = epoch < n_warm
        k = jax.random.fold_in(key, epoch)
        data, rst = dist.rollout(params, k, sev, warm)
        lr = lr_at(b, frac) * lr_mult * (~failed)
        p2, o2, st, ok = dist.update(params, opt_state, data, jax.random.fold_in(k, 1), jnp.asarray(lr, jnp.float32))
        ok = np.asarray(ok)
        for i in np.where(~ok & ~failed)[0]:
            nan_counts[i] += 1
            lr_mult[i] *= 0.5
            ctx.note(f'{tag}/{names[i]}: non-finite update at epoch {epoch} (#{nan_counts[i]}), skipped, lr halved')
            if nan_counts[i] > 3:
                failed[i] = True
                ctx.note(f'{tag}/{names[i]}: FAILED after >3 non-finite updates (frozen)')
        if dist.has_heads and epoch % diag_every == 0:
            dg = jax.tree.map(np.asarray, dist.diag(params, data))
        else:
            dg = None
        params, opt_state = p2, o2
        epoch += 1
        now = time.time()
        sps = dist.M * dist.N * dist.T / max(now - t_last, 1e-6)
        t_last = now
        st_h, rst_h = jax.tree.map(np.asarray, st), jax.tree.map(np.asarray, rst)
        rec = dict(epoch=epoch, tag=tag, sev=sev, warm=bool(warm), sps=sps,
                   models={n: {**{k2: float(v[i]) for k2, v in st_h.items()}, **{k2: float(v[i]) for k2, v in rst_h.items()},
                               'lr': float(lr[i])} for i, n in enumerate(names)})
        if dg is not None:
            rec['diag'] = {n: {k2: float(v[i]) for k2, v in dg.items()} for i, n in enumerate(names)}
        if epoch in COPYCAT_EPOCHS:
            copycat[str(epoch)] = {n: float(rst_h['online_mse'][i]) for i, n in enumerate(names)}
        ctx.metrics.write(**rec)
        if epoch % 10 == 1 or epoch == n_epochs:
            m0 = rec['models'][names[0]]
            ctx.log(f"{tag} epoch {epoch}/{n_epochs} sev={sev:.2f} {names[0]}: act={m0['act']:.4f} "
                    f"term={m0['term_rate']:.4f} online={m0['online_mse']:.4f} {sps:.0f} sps")
        if ctx.throttle.add(sps) and not alerted:
            alerted = True
            ctx.note(f'{tag}: ALERT throughput >40 % below its baseline for 15 min (training continues unchanged)')
        if epoch % sel_every == 0 or epoch == n_epochs:
            sel_store.save(tag, epoch, params)
        if ctx.ckpt.due():
            ctx.ckpt.save(epoch, dict(tag=tag, params=params, opt_state=opt_state, epoch=epoch, lr_mult=lr_mult,
                                      nan_counts=nan_counts, copycat=copycat))
    meta = dict(epoch=epoch, failed=[names[i] for i in np.where(failed)[0]], nan_counts=nan_counts.tolist(),
                copycat=copycat, models=[list(m) for m in dist.models],
                state=dict(opt_state=common.to_numpy(opt_state), lr_mult=lr_mult, nan_counts=nan_counts))
    common.save_params(final_path, params, meta)
    return meta
