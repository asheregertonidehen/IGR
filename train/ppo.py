"""Multi-model PPO for feed-forward (E) and recurrent (T) policies with an
asymmetric critic.

All M models of a program are vmapped: per-model parameters, Adam state, learning rate,
schedule values (severity, truth probability, imitation alpha), labeler parameters,
unsolved-drone label mask and NaN guard (a model with a non-finite update is restored and its
lr halved; after 3 such events it is frozen and reported FAILED while the others continue).

* Gaussian policy in pre-tanh space; executed action tanh(u); log-prob in u.
* Recurrent: rollouts are chunks of T steps; the carry at chunk start is stored and the update
  re-runs the policy over the chunk (truncated BPTT), resetting the carry after episode ends.
* Truncation at the episode limit bootstraps gamma * V(s') into the reward.
* Auto-reset draws from an episode pool sampled once per rollout. With `resample` (E)
  the pool's drones are drawn with probability proportional to (1 - normalized running return)^0.5,
  floored at 0.25 x uniform; the running return is a per-train-drone EMA of the per-step reward.
* Optional labeler (E or T) gives imitation targets with weight alpha, masked on the label
  teacher's unsolved drones; optional event-weighted representation losses (teacher).
Rewards are scaled by REWARD_SCALE (does not change the optimum).
"""
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models import critic as crit
from models import inputs as I
from sim import env as envm
from train import common, diagnostics, losses

REWARD_SCALE = 0.1
LOG2PI = jnp.log(2 * jnp.pi)
SCORE_EMA = 0.1
RESAMPLE_POWER = 0.5
RESAMPLE_FLOOR = 0.25


def gauss_logp(u, mean, log_std):
    z = (u - mean) * jnp.exp(-log_std)
    return jnp.sum(-0.5 * z * z - log_std - 0.5 * LOG2PI, axis=-1)


def make_optimizer(lr=1e-3):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adam)(learning_rate=lr))


def _set_lr(opt_state, lr):
    opt_state[1].hyperparams['learning_rate'] = jnp.asarray(lr, jnp.float32)
    return opt_state


def resample_probs(score, seen):
    """Drone sampling probabilities from the running per-drone return (higher return = easier)."""
    big = jnp.where(seen, score, jnp.inf)
    lo = jnp.min(big)
    hi = jnp.max(jnp.where(seen, score, -jnp.inf))
    norm = jnp.where(seen, (score - lo) / jnp.maximum(hi - lo, 1e-6), 0.0)   # unseen -> hardest
    p = (1.0 - norm) ** RESAMPLE_POWER + 1e-6
    p = p / jnp.sum(p)
    # floor at RESAMPLE_FLOOR x uniform, exactly: mixture with the uniform distribution
    return (1.0 - RESAMPLE_FLOOR) * p + RESAMPLE_FLOOR / score.shape[0]


class PPO:
    def __init__(self, arch, cfg, dset, spec, n_envs, T, hp, M, labeler=None, rep=None, resample=False,
                 ep_steps=500):
        """cfg: dict of [M] arrays (may be empty for E); labeler: (arch, lcfg dict of [M] arrays) or None;
        rep: dict of [M] arrays (pred, truth, pc, id, beta) or None."""
        self.arch, self.cfg, self.dset, self.spec = arch, cfg, dset, spec
        self.N, self.T, self.hp, self.M = n_envs, T, hp, M
        self.labeler, self.rep, self.resample = labeler, rep, resample
        self.n_drones = int(dset['mass'].shape[0])
        self.ep_steps = ep_steps
        self.ep_s = ep_steps * 0.01
        self.opt = make_optimizer(hp['lr'])
        self.rollout = jax.jit(jax.vmap(self._rollout1, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        self.update = jax.jit(jax.vmap(self._update1, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)))
        self.diag = jax.jit(jax.vmap(self._diag1, in_axes=(0, 0, 0, 0, 0)))

    # ---------------------------------------------------------------- helpers
    def _carry0(self, arch, n):
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), arch.init_carry())

    def init_params(self, keys, actor=None):
        """keys [M]; actor: stacked actor params [M] or None."""
        ka = jax.vmap(lambda k: jax.random.fold_in(k, 1))(keys)
        kc = jax.vmap(lambda k: jax.random.fold_in(k, 2))(keys)
        actor = actor if actor is not None else jax.vmap(self.arch.init)(ka)
        return dict(actor=actor, critic=jax.vmap(crit.init)(kc))

    def init_opt(self, params):
        return jax.vmap(self.opt.init)(params)

    def _pool(self, key, sev, p_truth, rs):
        if self.resample:
            p = resample_probs(rs['score'], rs['seen'])
            idx = jax.random.choice(jax.random.fold_in(key, 3), self.n_drones, (self.N,), p=p)
        else:
            idx = None
        return common.sample_pool(key, self.dset, self.spec, sev, self.N, p_truth, drone_idx=idx)

    def init_state(self, keys, sev, p_truth):
        """keys [M]; sev, p_truth [M]."""
        def one(key, sv, pt):
            rs = dict(score=jnp.zeros(self.n_drones), seen=jnp.zeros(self.n_drones, bool))
            env = jax.vmap(envm.reset)(self._pool(key, sv, pt, rs))
            # stagger episode clocks so resets desynchronize
            env = dict(env, k=jax.random.randint(jax.random.fold_in(key, 1), (self.N,), 0, self.ep_steps // 2))
            lab = self._carry0(self.labeler[0], self.N) if self.labeler else jnp.zeros(())
            return dict(env=env, carry=self._carry0(self.arch, self.N), lab=lab, **rs)
        return jax.jit(jax.vmap(one))(keys, jnp.asarray(sev, jnp.float32), jnp.asarray(p_truth, jnp.float32))

    # ---------------------------------------------------------------- rollout (one model)
    def _rollout1(self, params, lab_params, cfg, lcfg, rs, key, sev, p_truth, lab_mask, rep_w):
        key, kpool = jax.random.split(key)
        fresh = jax.vmap(envm.reset)(self._pool(kpool, sev, p_truth, rs))
        carry_init = self._carry0(self.arch, self.N)
        lab_init = self._carry0(self.labeler[0], self.N) if self.labeler else jnp.zeros(())
        gamma = self.hp['gamma']
        feat = lambda arch, c, o, pv: jax.vmap(lambda a, b: arch.features(a, b, c))(o, pv)
        stepf = lambda arch, p, c, carry, ch: jax.vmap(lambda cc, x: arch.step(p, c, cc, x))(carry, ch)

        def one(st, k):
            env = st['env']
            ka, ke = jax.random.split(k)
            o = env['obs']
            pv = jax.vmap(envm.privileged)(env)
            ch = feat(self.arch, cfg, o, pv)
            carry, mean, aux = stepf(self.arch, params['actor'], cfg, st['carry'], ch)
            log_std = params['actor']['log_std']
            u = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
            a = jnp.tanh(u)
            cx = jax.vmap(I.critic)(o, pv, env['t'] / self.ep_s)
            v = crit.value(params['critic'], cx)
            didx = env['ep']['drone_idx']
            rec = dict(lab_w=lab_mask[didx].astype(jnp.float32))
            lab = st['lab']
            if self.labeler:
                larch = self.labeler[0]
                lch = feat(larch, lcfg, o, pv)
                lab, lmean, _ = stepf(larch, lab_params, lcfg, lab, lch)
                rec['a_lab'] = jnp.tanh(lmean)
            env2, r, term = jax.vmap(envm.step)(env, a, jax.random.split(ke, self.N))
            trunc = env2['k'] >= self.ep_steps
            pv2 = jax.vmap(envm.privileged)(env2)
            v2 = crit.value(params['critic'], jax.vmap(I.critic)(env2['obs'], pv2, env2['t'] / self.ep_s))
            done = term | trunc
            r_adj = r * REWARD_SCALE + gamma * v2 * (trunc & ~term)
            rec.update(ch=ch, u=u, logp=gauss_logp(u, mean, log_std), v=v, r=r_adj, done=done,
                       term=term, a=a, cx=cx, raw_r=r, didx=didx)
            if self.rep is not None:
                rec.update(truth=jax.vmap(I.truth_targets)(pv), sig=jax.vmap(I.pred_signal)(o),
                           idt=jax.vmap(I.id_targets)(pv), pc=jax.vmap(I.pc_targets)(pv), ev=pv['event_active'])
            new = dict(st, env=common.tree_where(done, fresh, env2),
                       carry=common.tree_where(done, carry_init, carry),
                       lab=common.tree_where(done, lab_init, lab) if self.labeler else lab)
            return new, rec

        carry0 = rs['carry']
        rs, traj = jax.lax.scan(one, rs, jax.random.split(key, self.T))
        pvl = jax.vmap(envm.privileged)(rs['env'])
        v_last = crit.value(params['critic'], jax.vmap(I.critic)(rs['env']['obs'], pvl, rs['env']['t'] / self.ep_s))
        adv, ret = self._gae(traj['r'], traj['v'], traj['done'], v_last)
        traj.update(carry0=carry0, adv=adv, ret=ret)
        if self.resample:
            flat_d, flat_r = traj['didx'].reshape(-1), traj['raw_r'].reshape(-1)
            s = jax.ops.segment_sum(flat_r, flat_d, self.n_drones)
            c = jax.ops.segment_sum(jnp.ones_like(flat_r), flat_d, self.n_drones)
            m = s / jnp.maximum(c, 1.0)
            obs_ = c > 0
            score = jnp.where(obs_, jnp.where(rs['seen'], (1 - SCORE_EMA) * rs['score'] + SCORE_EMA * m, m), rs['score'])
            rs = dict(rs, score=score, seen=rs['seen'] | obs_)
        stats = dict(reward=jnp.mean(traj['raw_r']), term_rate=jnp.mean(traj['term']),
                     done_rate=jnp.mean(traj['done']), lab_frac=jnp.mean(traj['lab_w']))
        return rs, traj, stats

    def _gae(self, r, v, done, v_last):
        g, lam = self.hp['gamma'], self.hp['lam']

        def body(c, x):
            adv_n, v_n = c
            r_t, v_t, d_t = x
            nt = 1.0 - d_t.astype(jnp.float32)
            delta = r_t + g * v_n * nt - v_t
            adv = delta + g * lam * nt * adv_n
            return (adv, v_t), adv
        _, adv = jax.lax.scan(body, (jnp.zeros_like(v_last), v_last), (r, v, done), reverse=True)
        return adv, adv + v

    # ---------------------------------------------------------------- losses (one model)
    def _forward(self, actor, cfg, mb):
        arch = self.arch
        init = self._carry0(arch, mb['done'].shape[1])
        done_prev = jnp.concatenate([jnp.zeros_like(mb['done'][:1]), mb['done'][:-1]], 0)
        keep = ('h_rep', 'truth_head', 'id_head', 'pc_mu', 'pc_logvar') if self.rep is not None else ()

        def step(carry, x):
            ch, dp = x
            carry = common.tree_where(dp, init, carry)
            carry, mean, aux = jax.vmap(lambda c, z: arch.step(actor, cfg, c, z))(carry, ch)
            return carry, (mean, {k: aux[k] for k in keep if k in aux})
        _, (mean, outs) = jax.lax.scan(step, mb['carry0'], (mb['ch'], done_prev))
        return mean, outs

    def _control(self, params, mean, mb, alpha):
        log_std = params['actor']['log_std']
        logp = gauss_logp(mb['u'], mean, log_std)
        ratio = jnp.exp(logp - mb['logp'])
        adv = (mb['adv'] - mb['adv'].mean()) / (mb['adv'].std() + 1e-8)
        clip = self.hp['clip']
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv))
        out = dict(pg=pg, kl=jnp.mean(mb['logp'] - logp), clipfrac=jnp.mean(jnp.abs(ratio - 1) > clip),
                   std=jnp.mean(jnp.exp(log_std)))
        ctrl = self.hp.get('pg_coef', 1.0) * pg
        if self.labeler:
            lw = mb['lab_w']
            im = jnp.sum(jnp.mean((jnp.tanh(mean) - mb['a_lab']) ** 2, -1) * lw) / jnp.maximum(jnp.sum(lw), 1.0)
            ctrl = ctrl + alpha * im
            out['imitation'] = im
        return ctrl, out

    def _rep(self, params, outs, mb, rep_w):
        w = losses.event_weights(mb['ev'], rep_w['beta'])
        lr_ = losses.rep_losses(None, params['actor']['heads'], outs, mb, w)
        total = sum(rep_w[k] * v for k, v in lr_.items())
        return total, lr_

    def _loss1(self, params, mb, alpha, cfg, rep_w):
        mean, outs = self._forward(params['actor'], cfg, mb)
        ctrl, st = self._control(params, mean, mb, alpha)
        v = crit.value(params['critic'], mb['cx'])
        vf = 0.5 * jnp.mean((v - mb['ret']) ** 2)
        total = ctrl + self.hp['vf_coef'] * vf
        st['vf'] = vf
        if self.rep is not None:
            rt, lr_ = self._rep(params, outs, mb, rep_w)
            total = total + rt
            st.update({f'rep_{k}': x for k, x in lr_.items()})
        st['loss'] = total
        return total, st

    def _update1(self, params, opt_state, traj, key, alpha, lr, cfg, rep_w):
        n_mb, n_ep = self.hp['minibatches'], self.hp['epochs']
        mb_size = self.N // n_mb
        opt_state = _set_lr(opt_state, lr)

        def take(idx):
            return {k: (jax.tree.map(lambda x: x[idx], v) if k == 'carry0' else jax.tree.map(lambda x: x[:, idx], v))
                    for k, v in traj.items()}

        def epoch(c, k):
            def mb_step(c, idx):
                p, o = c
                (_, st), g = jax.value_and_grad(self._loss1, has_aux=True)(p, take(idx), alpha, cfg, rep_w)
                upd, o = self.opt.update(g, o, p)
                st['grad_norm'] = optax.tree.norm(g)
                return (optax.apply_updates(p, upd), o), st
            perm = jax.random.permutation(k, self.N).reshape(n_mb, mb_size)
            return jax.lax.scan(mb_step, c, perm)
        (p2, o2), st = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, n_ep))
        st = jax.tree.map(jnp.mean, st)
        ok = common.finite_tree(p2) & jnp.isfinite(st['loss'])
        p2 = jax.tree.map(lambda a, b: jnp.where(ok, a, b), p2, params)
        o2 = jax.tree.map(lambda a, b: jnp.where(ok, a, b), o2, opt_state)
        return p2, o2, st, ok

    def _diag1(self, params, traj, alpha, cfg, rep_w):
        n = min(self.N, 256)   # held-in batch: the first n envs of the rollout
        mb = {k: (jax.tree.map(lambda x: x[:n], v) if k == 'carry0' else jax.tree.map(lambda x: x[:, :n], v))
              for k, v in traj.items()}

        def ctrl_fn(p):
            mean, _ = self._forward(p['actor'], cfg, mb)
            return self._control(p, mean, mb, alpha)[0]

        def rep_fn(name):
            def f(p):
                _, outs = self._forward(p['actor'], cfg, mb)
                return self._rep(p, outs, mb, rep_w)[1][name]
            return f
        mean, outs = self._forward(params['actor'], cfg, mb)
        names = ['pred', 'truth', 'id'] + (['pc'] if 'pc_mu' in outs else [])
        out = diagnostics.cosines(lambda p: ctrl_fn(dict(params, actor=p)),
                                  {k: (lambda p, f=rep_fn(k): f(dict(params, actor=p))) for k in names},
                                  params['actor'])
        out.update(diagnostics.head_r2(outs, mb))
        ctrl = self._control(params, mean, mb, alpha)[0]
        _, lr_ = self._rep(params, outs, mb, rep_w)
        out.update({f'rel_{k}': rep_w[k] * v / jnp.maximum(jnp.abs(ctrl), 1e-8) for k, v in lr_.items()})
        return out


# ------------------------------------------------------------------ host loop
def _host(tree):
    return jax.tree.map(lambda x: np.asarray(x), tree)


def train_loop(ctx, ppo, params, total_steps, schedule, key, names, lab_params=None, lab_mask=None, tag='ppo',
               stage_steps=None, start_step=0, opt_state=None, sel_every=None, sel_store=None, failed=None):
    """Run PPO updates until `total_steps` env steps per model.

    schedule(frac) -> dict(sev, p_truth, alpha, lr), each a scalar or [M] array; frac = progress over
    the stage. Protections: per-model NaN guard (restore, lr x0.5, frozen + FAILED after 3); resume
    checkpoints every ctx.ckpt.every_s; selection checkpoints every `sel_every` env steps (all kept).
    No time-based truncation. Returns (params, opt_state, step, info)."""
    M = ppo.M
    stage_steps = stage_steps or total_steps
    steps_per = ppo.N * ppo.T
    final_path = ctx.job_dir / f'stage_{tag}.pkl'
    if final_path.exists():  # stage already completed in an earlier (interrupted) run
        d = pickle.loads(final_path.read_bytes())
        ctx.log(f'{tag}: stage already complete ({d["step"]} steps), reusing')
        return (jax.tree.map(jnp.asarray, d['params']), jax.tree.map(jnp.asarray, d['opt_state']), d['step'], d['info'])
    lab_params = lab_params if lab_params is not None else jnp.zeros((M,))
    lab_mask = jnp.asarray(lab_mask if lab_mask is not None else np.ones((M, ppo.n_drones), bool))
    lcfg = ppo.labeler[1] if ppo.labeler else {}
    rep_w = ppo.rep if ppo.rep is not None else {'beta': jnp.zeros(M)}
    opt_state = opt_state if opt_state is not None else ppo.init_opt(params)
    step, lr_mult, nan_counts = start_step, np.ones(M), np.zeros(M, int)
    failed = np.zeros(M, bool) if failed is None else np.asarray(failed, bool).copy()
    key, k0 = jax.random.split(key)
    s0 = _sched(schedule, min(1.0, step / stage_steps), M)
    rs = ppo.init_state(jax.random.split(k0, M), s0['sev'], s0['p_truth'])
    ck = ctx.ckpt.latest()
    if ck is not None and ck.get('tag') == tag and ck['step'] > step:
        params = jax.tree.map(jnp.asarray, ck['params'])
        opt_state = jax.tree.map(jnp.asarray, ck['opt_state'])
        step, lr_mult, nan_counts, failed = ck['step'], np.asarray(ck['lr_mult']), np.asarray(ck['nan_counts']), \
            np.asarray(ck['failed'])
        key = jnp.asarray(ck['key'], dtype=jnp.uint32)
        if 'score' in ck:
            rs = dict(rs, score=jnp.asarray(ck['score']), seen=jnp.asarray(ck['seen']))
        ctx.log(f'resumed {tag} at {step} env steps')
    next_sel = (step // sel_every + 1) * sel_every if sel_every else None
    t_last, n_upd, alerted = time.time(), 0, False
    while step < total_steps:
        sc = _sched(schedule, min(1.0, step / stage_steps), M)
        key, k1, k2 = jax.random.split(key, 3)
        rs2, traj, rst = ppo.rollout(params, lab_params, ppo.cfg, lcfg, rs, jax.random.split(k1, M), sc['sev'],
                                     sc['p_truth'], lab_mask, rep_w)
        lr = jnp.asarray(sc['lr'] * lr_mult * (~failed), jnp.float32)
        p2, o2, ust, ok = ppo.update(params, opt_state, traj, jax.random.split(k2, M), sc['alpha'], lr, ppo.cfg, rep_w)
        ok = np.asarray(ok)
        rst_h = _host(rst)
        for i in np.where(~ok & ~failed)[0]:
            nan_counts[i] += 1
            lr_mult[i] *= 0.5
            ctx.note(f'{tag}/{names[i]}: non-finite update at {step} (#{nan_counts[i]}); skipped, lr halved')
            if nan_counts[i] > 3:
                failed[i] = True
                ctx.note(f'{tag}/{names[i]}: FAILED after >3 non-finite updates (frozen)')
        bad_roll = ~np.isfinite(rst_h['reward'])
        params, opt_state = p2, o2
        rs = common.tree_where(jnp.asarray(~bad_roll), rs2, rs) if bad_roll.any() else rs2
        step += steps_per
        n_upd += 1
        now = time.time()
        sps = M * steps_per / max(now - t_last, 1e-6)
        t_last = now
        ust_h = _host(ust)
        rec = dict(step=step, tag=tag, sps=sps,
                   models={n: {**{k: float(v[i]) for k, v in ust_h.items()}, **{k: float(v[i]) for k, v in rst_h.items()},
                               'sev': float(np.broadcast_to(sc['sev'], (M,))[i]),
                               'p_truth': float(np.broadcast_to(sc['p_truth'], (M,))[i]),
                               'alpha': float(np.broadcast_to(sc['alpha'], (M,))[i]), 'lr': float(lr[i])}
                           for i, n in enumerate(names)})
        if ppo.rep is not None and n_upd % max(1, diagnostics.DIAG_EVERY_UPDATES // (ppo.hp['epochs'] * ppo.hp['minibatches'])) == 1:
            dg = _host(ppo.diag(params, traj, sc_alpha(sc, M), ppo.cfg, rep_w))
            rec['diag'] = {n: {k: float(v[i]) for k, v in dg.items()} for i, n in enumerate(names)}
        ctx.metrics.write(**rec)
        if n_upd % 10 == 1:
            m0 = rec['models'][names[0]]
            ctx.log(f"{tag} {step/1e6:.2f}M/{total_steps/1e6:.2f}M {names[0]}: r={m0['reward']:.3f} "
                    f"term={m0['term_rate']:.4f} vf={m0['vf']:.3f} std={m0['std']:.3f} | {sps:.0f} sps (all models)")
        if ctx.throttle.add(sps) and not alerted:
            alerted = True
            ctx.note(f'{tag}: ALERT throughput >40 % below its baseline for 15 min (training continues unchanged)')
        if sel_every and step >= next_sel:
            sel_store.save(tag, step, params['actor'])
            next_sel += sel_every
        if ctx.ckpt.due():
            _save_ck(ctx, tag, params, opt_state, step, lr_mult, nan_counts, failed, key, rs)
    if sel_store is not None and sel_every and (not sel_store.steps(tag) or sel_store.steps(tag)[-1] != step):
        sel_store.save(tag, step, params['actor'])
    info = dict(nan_counts=nan_counts.tolist(), failed=[names[i] for i in np.where(failed)[0]], failed_mask=failed.tolist())
    common.atomic_write_bytes(final_path, pickle.dumps(common.to_numpy(
        dict(params=params, opt_state=opt_state, step=step, info=info))))
    return params, opt_state, step, info


def sc_alpha(sc, M):
    return jnp.broadcast_to(jnp.asarray(sc['alpha'], jnp.float32), (M,))


def _sched(schedule, frac, M):
    sc = schedule(frac)
    return {k: jnp.broadcast_to(jnp.asarray(v, jnp.float32), (M,)) for k, v in sc.items()}


def _save_ck(ctx, tag, params, opt_state, step, lr_mult, nan_counts, failed, key, rs):
    d = dict(tag=tag, params=params, opt_state=opt_state, step=step, lr_mult=lr_mult, nan_counts=nan_counts,
             failed=failed, key=np.asarray(key))
    if 'score' in rs:
        d.update(score=rs['score'], seen=rs['seen'])
    ctx.ckpt.save(step, d)
