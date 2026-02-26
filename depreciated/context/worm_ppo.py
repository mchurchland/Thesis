# ray_reservoir_ppo_ws_video.py
import os
# Prevent BLAS from oversubscribing threads inside each Ray worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
import shutil
import ray

# ==================== utils ====================
class RunningNorm:
    def __init__(self, n, eps=1e-8):
        self.m = np.zeros(n, np.float32)
        self.s2 = np.ones(n, np.float32)
        self.n = eps; self.eps = eps
    def __call__(self, x, update=True):
        x = np.asarray(x, np.float32)
        if update:
            self.n += 1.0
            d = x - self.m
            self.m += d / self.n
            self.s2 += d * (x - self.m)
        std = np.sqrt(self.s2 / self.n) + self.eps
        return np.clip((x - self.m) / std, -5.0, 5.0)

def atanh_clip(a):
    a = np.clip(a, -0.999999, 0.999999)
    return 0.5 * (np.log1p(a) - np.log1p(-a))

def logp_tanh_gauss(a, mu, log_std):
    u = atanh_clip(a)
    std = np.exp(log_std)
    lp = -0.5 * np.sum(((u - mu)/std)**2 + 2*log_std + np.log(2*np.pi), axis=-1)
    lp -= np.sum(np.log(1 - a*a + 1e-6), axis=-1)  # tanh correction
    return lp, u

# ==================== Watts–Strogatz reservoir ====================
def ws_adjacency(d, k, p, rng):
    assert k % 2 == 0 and 0 <= p <= 1.0 and k < d
    A = np.zeros((d, d), dtype=bool)
    # ring lattice
    for i in range(d):
        for j in range(1, k//2 + 1):
            A[i, (i+j) % d] = True
            A[i, (i-j) % d] = True
    # WS rewiring
    for i in range(d):
        for j in range(1, k//2 + 1):
            if rng.random() < p:
                old = (i + j) % d
                if not A[i, old]:
                    continue
                A[i, old] = False
                A[old, i] = False
                candidates = np.where(~A[i] & (np.arange(d) != i))[0]
                if candidates.size == 0:
                    A[i, old] = True; A[old, i] = True
                else:
                    new = rng.choice(candidates)
                    A[i, new] = True; A[new, i] = True
    np.fill_diagonal(A, False)
    return A

def build_ws_reservoir(d, obs_dim, k=20, p=0.1, sr=0.9, scale_in=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = ws_adjacency(d, k, p, rng)
    mask = A.astype(np.float32)
    Wres = rng.standard_normal((d, d)).astype(np.float32) * mask
    eigmax = max(1e-6, float(np.max(np.abs(np.linalg.eigvals(Wres)))))
    Wres = (sr / eigmax) * Wres.real.astype(np.float32)
    Win = (scale_in * rng.standard_normal((d, obs_dim))).astype(np.float32)
    return Wres, Win

# ==================== eval (random-start) ====================
def eval_random_start(env, Wmu_eval, step_res, norm, d,
                      phi_gain=1.0, obs_noise_std=0.0, episodes=10, seed=10_000):
    rets = []
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    for k in range(episodes):
        z = np.zeros(d, np.float32)
        obs, _ = env.reset(seed=seed + k)
        done=False; ep_ret=0.0; steps=0
        while not done and steps < 2000:
            if obs_noise_std > 0:
                obs = obs + rng.normal(0.0, obs_noise_std, size=obs_dim).astype(np.float32)
            x = norm(obs, update=False)
            z = step_res(z, x)
            ph = np.concatenate([z * phi_gain, [1.0]], dtype=np.float32)
            mu = Wmu_eval @ ph
            a  = np.tanh(mu)
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += float(r); done = term or trunc; steps += 1
        rets.append(ep_ret)
    return float(np.mean(rets)), float(np.std(rets))

# ==================== PPO over reservoir features ====================
def train_swimmer_ppo(
    episodes=200,
    d=600, leak=0.3,
    gamma=0.99, lam=0.95,
    T=4096, K=4, MB=512, clip_eps=0.2,
    alpha_pi=3e-3, alpha_v=3e-4,
    entropy_coef=0.01, value_coef=0.25,
    l2_v=1e-5, l2_pi=1e-4,
    v_clip=0.2, target_kl=0.015,
    log_std_min=-1.2, log_std_max=-0.2, init_log_std=-0.6,
    sr=0.9, ws_k=20, ws_p=0.1, scale_in=1.0,
    grad_clip=2.0, washout_steps=50,
    ema_eval=True, ema_tau=0.01,
    eval_every=10, eval_rand_eps=10,
    phi_gain=5.0, logstd_decay=0.995,
    adv_clip=5.0,
    obs_noise_std=0.0,
    seed=0
):
    rng = np.random.default_rng(seed)
    env = gym.make("Swimmer-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Reservoir
    Wres, Win = build_ws_reservoir(d, obs_dim, k=ws_k, p=ws_p, sr=sr, scale_in=scale_in, seed=seed)
    def step_res(z, x):
        h = np.tanh(Wres @ z + Win @ x)
        return (1 - leak) * z + leak * h

    # Heads
    v = np.zeros(d + 1, np.float32)
    Wmu = np.zeros((act_dim, d + 1), np.float32)
    Wmu_ema = Wmu.copy()
    log_std = np.full(act_dim, init_log_std, np.float32)
    norm = RunningNorm(obs_dim)

    # rollout
    def collect_rollout(T):
        PH=[]; A=[]; R=[]; Vt=[]; LP=[]; DONE=[]
        steps=0; z = np.zeros(d, np.float32); done=True; obs=None
        obs_rng = np.random.default_rng(seed + 12345)
        ph2 = None; done_flag = True
        while steps < T:
            if done:
                z = np.zeros(d, np.float32)
                obs, _ = env.reset(seed=seed + steps)
                # washout
                for _ in range(washout_steps):
                    xw = obs.copy()
                    if obs_noise_std > 0:
                        xw = xw + obs_rng.normal(0.0, obs_noise_std, size=obs_dim).astype(np.float32)
                    xw = norm(xw)
                    z = step_res(z, xw)
                    phw = np.concatenate([z * phi_gain, [1.0]], dtype=np.float32)
                    muw = Wmu @ phw
                    aw  = np.tanh(muw)
                    obs, _, term, trunc, _ = env.step(aw)
                    if term or trunc:
                        obs, _ = env.reset(seed=seed + steps + 1)
                        z = np.zeros(d, np.float32)

            x = obs.copy()
            if obs_noise_std > 0:
                x = x + obs_rng.normal(0.0, obs_noise_std, size=obs_dim).astype(np.float32)
            x = norm(x)
            z = step_res(z, x)
            ph = np.concatenate([z * phi_gain, [1.0]], dtype=np.float32)
            mu = Wmu @ ph

            std = np.exp(log_std)
            eps = rng.standard_normal(act_dim).astype(np.float32)
            u = mu + std * eps
            a = np.tanh(u)
            lp, _ = logp_tanh_gauss(a, mu, log_std)

            v_t = float(v @ ph)
            obs2, r, term, trunc, _ = env.step(a)
            done_step = bool(term or trunc)

            x2 = obs2.copy()
            if obs_noise_std > 0:
                x2 = x2 + obs_rng.normal(0.0, obs_noise_std, size=obs_dim).astype(np.float32)
            x2 = norm(x2)
            z2 = step_res(z, x2)
            ph2 = np.concatenate([z2 * phi_gain, [1.0]], dtype=np.float32)

            PH.append(ph); A.append(a); R.append(float(r)); Vt.append(v_t); LP.append(float(lp)); DONE.append(done_step)

            obs = obs2; z = z2; done = done_step; steps += 1
            done_flag = done_step

        return (np.stack(PH),
                np.stack(A),
                np.array(R, np.float32),
                np.array(Vt, np.float32),
                np.array(LP, np.float32),
                np.array(DONE, np.bool_),
                ph2, bool(done_flag))

    def gae_advantages(R, V, DONE, last_v):
        Tn = len(R)
        adv = np.zeros(Tn, np.float32)
        ret = np.zeros(Tn, np.float32)
        gae = 0.0; next_v = last_v
        for t in reversed(range(Tn)):
            delta = R[t] + (0 if DONE[t] else gamma*next_v) - V[t]
            gae = delta + gamma*lam*(0 if DONE[t] else 1.0)*gae
            adv[t] = gae
            ret[t] = adv[t] + V[t]
            next_v = V[t]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if adv_clip is not None:
            adv = np.clip(adv, -adv_clip, adv_clip)
        return adv, ret

    eval_series = []
    for ep in range(1, episodes+1):
        PH, A, R, V_old, OLDLP, DONE, last_ph, last_done = collect_rollout(T)
        last_v = 0.0 if last_done or last_ph is None else float(v @ last_ph)
        ADV, RET = gae_advantages(R, V_old, DONE, last_v)

        # PPO updates
        N = len(RET); idx_all = np.arange(N)
        agg_kl = []
        for _ in range(K):
            np.random.shuffle(idx_all)
            for j in range(0, N, MB):
                mb = idx_all[j:j+MB]
                ph_mb = PH[mb]; a_mb = A[mb]; adv_mb = ADV[mb]; ret_mb = RET[mb]
                oldlp_mb = OLDLP[mb]; v_old_mb = V_old[mb]

                mu_mb = (Wmu @ ph_mb.T).T
                lp_mb, u_mb = logp_tanh_gauss(a_mb, mu_mb, log_std)

                ratio   = np.exp(lp_mb - oldlp_mb)
                ratio_c = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                w_ratio = np.where((adv_mb > 0) & (ratio > 1.0 + clip_eps), ratio_c,
                         np.where((adv_mb < 0) & (ratio < 1.0 - clip_eps), ratio_c, ratio))

                std = np.exp(log_std)
                grad_logp_mu = (u_mb - mu_mb) / (std**2)
                dWmu = ((w_ratio * adv_mb)[:, None] * grad_logp_mu).T @ ph_mb / max(1, len(mb))
                dWmu -= l2_pi * Wmu

                v_pred = (v @ ph_mb.T)
                v_pred_clip = v_old_mb + np.clip(v_pred - v_old_mb, -v_clip, v_clip)
                mse_unclipped = (v_pred - ret_mb)**2
                mse_clipped   = (v_pred_clip - ret_mb)**2
                use_clip = (mse_clipped > mse_unclipped).astype(np.float32)
                err = use_clip * (v_pred - ret_mb) + (1.0 - use_clip) * (v_pred_clip - ret_mb)
                dv  = (2.0 * (err[None, :] @ ph_mb)).ravel() / max(1, len(mb))
                dv += l2_v * v

                # grad clip
                gn_pi = float(np.linalg.norm(dWmu))
                gn_v  = float(np.linalg.norm(dv))
                if gn_pi > grad_clip: dWmu *= grad_clip / (gn_pi + 1e-8)
                if gn_v  > grad_clip: dv   *= grad_clip / (gn_v  + 1e-8)

                Wmu += alpha_pi * dWmu
                v   -= (alpha_v * value_coef) * dv

                # entropy nudge
                log_std = np.clip(log_std + alpha_pi * (entropy_coef * np.ones_like(log_std)),
                                  log_std_min, log_std_max)

                kl = float(np.mean(oldlp_mb - lp_mb))
                agg_kl.append(kl)

            if len(agg_kl) > 0 and np.mean(agg_kl[-max(1, N//MB):]) > 2.5*target_kl:
                break

        # adapt LR
        mean_kl = float(np.mean(agg_kl)) if agg_kl else 0.0
        if mean_kl < 0.3 * target_kl:
            alpha_pi = min(alpha_pi * 1.5, 1e-2)
        elif mean_kl > 2.0 * target_kl:
            alpha_pi = max(alpha_pi * 0.5, 5e-4)

        Wmu_ema = ema_tau * Wmu + (1 - ema_tau) * Wmu_ema if ema_eval else Wmu
        log_std = np.clip(log_std * logstd_decay, log_std_min, log_std_max)

        if ep % eval_every == 0:
            mean_rand, _ = eval_random_start(
                env, Wmu_ema, step_res, norm, d,
                phi_gain=phi_gain, obs_noise_std=obs_noise_std,
                episodes=eval_rand_eps, seed=10_000+ep)
            eval_series.append(mean_rand)

    env.close()
    return np.asarray(eval_series, dtype=np.float32)  # [n_eval_points]

# ==================== rolling mean ====================
def rolling_last_k_mean(series, k=10):
    s = np.asarray(series, float)
    if s.size == 0:
        return s
    out = np.zeros_like(s, float)
    csum = np.cumsum(s)
    for t in range(s.size):
        a = max(0, t - (k - 1)); b = t
        total = csum[b] - (csum[a-1] if a > 0 else 0.0)
        out[t] = total / (b - a + 1)
    return out

# ==================== Ray worker ====================
@ray.remote(num_cpus=1)
def _run_cell_ray(p, sigma, seed, episodes, eval_every, ws_k, d, T):
    return train_swimmer_ppo(
        episodes=episodes, T=T, d=d,
        ws_k=ws_k, ws_p=float(p),
        obs_noise_std=float(sigma),
        eval_every=eval_every,
        seed=int(seed)
    )

# ==================== sweep & collect (Ray) ====================
def sweep_collect_timeseries_ray(
    p_list, noise_list, seeds=(0,1),
    episodes=600, eval_every=10, ws_k=40, d=600, T=4096,
    num_cpus=10
):
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True, include_dashboard=False)

    nP = len(p_list); nN = len(noise_list)
    n_eval = episodes // eval_every

    # Launch all tasks
    jobs = []
    meta = []  # (yi, xi, seed_index)
    for yi, sigma in enumerate(noise_list):
        for xi, p in enumerate(p_list):
            for si, s in enumerate(seeds):
                jid = _run_cell_ray.remote(p, sigma, s, episodes, eval_every, ws_k, d, T)
                jobs.append(jid)
                meta.append((yi, xi, si))

    # Collect
    raw = [[[] for _ in range(nP)] for __ in range(nN)]
    for jid, (yi, xi, si) in zip(jobs, meta):
        series = ray.get(jid)
        # pad/truncate
        if len(series) < n_eval:
            pad = np.full(n_eval - len(series), np.nan, np.float32)
            series = np.concatenate([series, pad])
        elif len(series) > n_eval:
            series = series[:n_eval]
        raw[yi][xi].append(series.astype(np.float32))
    ray.shutdown()

    # Build cube with rolling-last-10 mean averaged across seeds
    cube = np.full((nN, nP, n_eval), np.nan, np.float32)
    for yi in range(nN):
        for xi in range(nP):
            seed_series = raw[yi][xi]
            if not seed_series:
                continue
            rolling_stack = [rolling_last_k_mean(ss, k=10) for ss in seed_series]
            rolling_stack = np.stack(rolling_stack, axis=0)  # [S, n_eval]
            cube[yi, xi] = np.nanmean(rolling_stack, axis=0)

    eval_times = np.arange(eval_every, episodes + 1, eval_every, dtype=int)
    return eval_times, cube

# ==================== animation ====================
def animate_heatmap(eval_times, cube, p_list, noise_list,
                    vmin=None, vmax=None, out_mp4="grid_evolution.mp4", out_gif="grid_evolution.gif"):
    nN, nP, n_eval = cube.shape
    if vmin is None: vmin = np.nanmin(cube)
    if vmax is None: vmax = np.nanmax(cube)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    im = ax.imshow(cube[:, :, 0], origin='lower', aspect='auto',
                   extent=[min(p_list), max(p_list), min(noise_list), max(noise_list)],
                   vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im)
    cbar.set_label("Eval return (rolling mean of last 10)")
    ax.set_xlabel("WS rewiring probability p")
    ax.set_ylabel("Observation noise σ")
    ttl = ax.set_title(f"Generation (episode) = {eval_times[0]}")
    ax.set_xticks(p_list); ax.set_yticks(noise_list)

    def update(frame):
        im.set_data(cube[:, :, frame])
        ttl.set_text(f"Generation (episode) = {eval_times[frame]}")
        return [im, ttl]

    anim = animation.FuncAnimation(fig, update, frames=n_eval, interval=500, blit=True)

    saved_any = False
    if shutil.which("ffmpeg") is not None:
        Writer = animation.FFMpegWriter
        writer = Writer(fps=4, metadata=dict(artist='reservoir-ppo'), bitrate=1800)
        anim.save(out_mp4, writer=writer)
        print(f"Saved video to {out_mp4}")
        saved_any = True
    elif shutil.which("magick") is not None or shutil.which("convert") is not None:
        anim.save(out_gif, writer='imagemagick', fps=4)
        print(f"Saved GIF to {out_gif}")
        saved_any = True
    else:
        print("No ffmpeg/ImageMagick found; not saving animation. (Install ffmpeg for MP4.)")

    im.set_data(cube[:, :, -1])
    ttl.set_text(f"Generation (episode) = {eval_times[-1]}")
    fig.tight_layout()
    fig.savefig("grid_lastframe.png", dpi=150)
    print("Saved last frame to grid_lastframe.png")
    plt.close(fig)
    return saved_any

# ==================== main ====================
if __name__ == "__main__":
    # grid
    p_list = np.round(np.linspace(0.0, 0.6, 5), 3)
    noise_list = np.round(np.linspace(0.0, 0.2, 5), 3)

    # training budget
    episodes = 1000
    eval_every = 10
    T = 8192
    d = 600
    ws_k = 60        # even, < d
    seeds = (0, 1)   # ↑ for more smoothing

    # run on 10 CPU workers
    eval_times, cube = sweep_collect_timeseries_ray(
        p_list=p_list,
        noise_list=noise_list,
        seeds=seeds,
        episodes=episodes,
        eval_every=eval_every,
        ws_k=ws_k,
        d=d,
        T=T,
        num_cpus=10
    )

    # make animation
    animate_heatmap(eval_times, cube, p_list, noise_list)
