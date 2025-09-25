# pip install "gymnasium[mujoco]" numpy matplotlib
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

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
    """
    Undirected WS ring-lattice with degree k (even), rewiring prob p.
    Returns boolean adjacency (no self-loops). We later assign weights and scale spectral radius.
    """
    assert k % 2 == 0 and 0 <= p <= 1.0 and k < d
    A = np.zeros((d, d), dtype=bool)
    # ring lattice
    for i in range(d):
        for j in range(1, k//2 + 1):
            A[i, (i+j) % d] = True
            A[i, (i-j) % d] = True
    # rewire each "right" edge (i -> i+j) to a random node (WS style)
    for i in range(d):
        for j in range(1, k//2 + 1):
            if rng.random() < p:
                # remove edge i-(i+j)
                old = (i + j) % d
                A[i, old] = False
                A[old, i] = False
                # pick new target not i and not existing neighbor
                candidates = np.where(~A[i] & (np.arange(d) != i))[0]
                if len(candidates) == 0:
                    # fallback: restore original
                    A[i, old] = True; A[old, i] = True
                else:
                    new = rng.choice(candidates)
                    A[i, new] = True; A[new, i] = True
    np.fill_diagonal(A, False)
    return A

def build_ws_reservoir(d, obs_dim, k=20, p=0.1, sr=0.9, scale_in=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = ws_adjacency(d, k, p, rng)
    # assign symmetric Gaussian weights on existing edges (could also choose asymmetric)
    Wres = np.zeros((d, d), np.float32)
    mask = A.astype(np.float32)
    Wres = rng.standard_normal((d, d)).astype(np.float32) * mask
    # scale spectral radius
    eigmax = max(1e-6, float(np.max(np.abs(np.linalg.eigvals(Wres)))))
    Wres = (sr / eigmax) * Wres.real.astype(np.float32)
    Win = (scale_in * rng.standard_normal((d, obs_dim))).astype(np.float32)
    return Wres, Win

# ==================== eval helpers (UNWRAPPED Mujoco) ====================
def snapshot_initial_state(env):
    env.reset(seed=123)
    qpos0 = env.unwrapped.data.qpos.copy()
    qvel0 = env.unwrapped.data.qvel.copy()
    return qpos0, qvel0

def _obs_unwrapped(env):
    return env.unwrapped._get_obs()

def eval_fixed_start(env, Wmu_eval, step_res, norm, d, qpos0, qvel0,
                     phi_gain=1.0, obs_noise_std=0.0, episodes=5, seed=123):
    rets = []
    rng = np.random.default_rng(seed)
    obs_dim = env.observation_space.shape[0]
    for _ in range(episodes):
        env.reset(seed=seed)
        env.unwrapped.set_state(qpos0.copy(), qvel0.copy())
        obs = _obs_unwrapped(env)
        z = np.zeros(d, np.float32); done=False; ep_ret=0.0; steps=0
        while not done and steps < 2000:
            # add observation noise only to the input (not to env state)
            if obs_noise_std > 0:
                obs = obs + rng.normal(0.0, obs_noise_std, size=obs_dim).astype(np.float32)
            x = norm(obs, update=False)
            z = step_res(z, x)
            ph = np.concatenate([z * phi_gain, [1.0]], dtype=np.float32)
            mu = Wmu_eval @ ph
            a  = np.tanh(mu)  # deterministic eval
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += float(r); done = term or trunc; steps += 1
        rets.append(ep_ret)
    return float(np.mean(rets)), float(np.std(rets))

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
            a  = np.tanh(mu)  # deterministic eval
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
    # reservoir params
    sr=0.9, ws_k=20, ws_p=0.1, scale_in=1.0,
    # training niceties
    grad_clip=2.0, washout_steps=50,
    ema_eval=True, ema_tau=0.01,
    eval_every=10, eval_fixed_eps=10, eval_rand_eps=20,
    phi_gain=5.0, logstd_decay=0.995,
    adv_clip=5.0,
    # noise
    obs_noise_std=0.0,
    seed=0
):
    rng = np.random.default_rng(seed)
    env = gym.make("Swimmer-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Reservoir (WS)
    Wres, Win = build_ws_reservoir(d, obs_dim, k=ws_k, p=ws_p, sr=sr, scale_in=scale_in, seed=seed)
    def step_res(z, x):
        h = np.tanh(Wres @ z + Win @ x)
        return (1 - leak) * z + leak * h

    # Heads
    v = np.zeros(d + 1, np.float32)                 # critic weights
    Wmu = np.zeros((act_dim, d + 1), np.float32)    # actor (mean)
    Wmu_ema = Wmu.copy()
    log_std = np.full(act_dim, init_log_std, np.float32)

    norm = RunningNorm(obs_dim)
    qpos0, qvel0 = snapshot_initial_state(env)
    ret_win = deque(maxlen=100)

    hist_ep, hist_eval_rand = [], []  # we’ll base “final” metric on eval_rand
    # rollout
    def collect_rollout(T):
        PH=[]; A=[]; R=[]; Vt=[]; LP=[]; DONE=[]
        steps=0; z = np.zeros(d, np.float32); done=True; obs=None
        obs_rng = np.random.default_rng(seed + 12345)
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

            # next features (for bootstrap only)
            x2 = obs2.copy()
            if obs_noise_std > 0:
                x2 = x2 + obs_rng.normal(0.0, obs_noise_std, size=obs_dim).astype(np.float32)
            x2 = norm(x2)
            z2 = step_res(z, x2)
            ph2 = np.concatenate([z2 * phi_gain, [1.0]], dtype=np.float32)

            PH.append(ph); A.append(a); R.append(float(r)); Vt.append(v_t); LP.append(float(lp)); DONE.append(done_step)

            obs = obs2; z = z2; done = done_step; steps += 1

        return (np.stack(PH),
                np.stack(A),
                np.array(R, np.float32),
                np.array(Vt, np.float32),       # v_old
                np.array(LP, np.float32),
                np.array(DONE, np.bool_),
                ph2, bool(done))

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

    for ep in range(1, episodes+1):
        PH, A, R, V_old, OLDLP, DONE, last_ph, last_done = collect_rollout(T)
        last_v = 0.0 if last_done or last_ph is None else float(v @ last_ph)
        ADV, RET = gae_advantages(R, V_old, DONE, last_v)

        # PPO updates
        N = len(RET); idx_all = np.arange(N)
        agg_kl = []
        for epoch in range(K):
            np.random.shuffle(idx_all)
            for j in range(0, N, MB):
                mb = idx_all[j:j+MB]
                ph_mb = PH[mb]
                a_mb  = A[mb]
                adv_mb = ADV[mb]
                ret_mb = RET[mb]
                oldlp_mb = OLDLP[mb]
                v_old_mb = V_old[mb]

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

                # clip gradients
                gn_pi = float(np.linalg.norm(dWmu))
                gn_v  = float(np.linalg.norm(dv))
                if gn_pi > grad_clip: dWmu *= grad_clip / (gn_pi + 1e-8)
                if gn_v  > grad_clip: dv   *= grad_clip / (gn_v  + 1e-8)

                Wmu += alpha_pi * dWmu
                v   -= (alpha_v * value_coef) * dv

                # entropy -> widen slightly
                log_std = np.clip(log_std + alpha_pi * (entropy_coef * np.ones_like(log_std)), log_std_min, log_std_max)

                kl = float(np.mean(oldlp_mb - lp_mb))
                agg_kl.append(kl)

            # early stop on big KL
            if np.mean(agg_kl[-(N//MB+1):]) > 2.5*target_kl:
                break

        # adapt actor LR
        mean_kl = float(np.mean(agg_kl)) if agg_kl else 0.0
        if mean_kl < 0.3 * target_kl:
            alpha_pi = min(alpha_pi * 1.5, 1e-2)
        elif mean_kl > 2.0 * target_kl:
            alpha_pi = max(alpha_pi * 0.5, 5e-4)

        # EMA readout for eval stability
        Wmu_ema = ema_tau * Wmu + (1 - ema_tau) * Wmu_ema if ema_eval else Wmu
        # decay exploration
        log_std = np.clip(log_std * logstd_decay, log_std_min, log_std_max)

        # periodic eval (random-start; we’ll use this series for “final”)
        if ep % eval_every == 0:
            mean_rand, std_rand = eval_random_start(
                env, Wmu_ema, step_res, norm, d,
                phi_gain=phi_gain, obs_noise_std=obs_noise_std,
                episodes=eval_rand_eps, seed=10_000+ep)
            hist_ep.append(ep)
            hist_eval_rand.append(mean_rand)

    env.close()
    return dict(history=dict(ep=np.array(hist_ep), eval_rand=np.array(hist_eval_rand)),
                Wmu=Wmu, v=v, log_std=log_std)

# ==================== summarizer ====================
def final_eval_mean(history_eval, window_frac=0.1):
    y = np.asarray(history_eval, float)
    mask = ~np.isnan(y)
    y = y[mask]
    if len(y) == 0:
        return np.nan
    w = max(3, int(len(y) * window_frac))  # last 10% (>=3 points)
    return float(np.mean(y[-w:]))

# ==================== sweep & heatmap ====================
def sweep_heatmap(
    p_list, noise_list, seeds=(0,1),  # increase seeds for robustness
    episodes=150, T=4096, ws_k=20, d=400,
    eval_every=10
):
    # grid results: rows = noise (y), cols = p (x)
    heat = np.full((len(noise_list), len(p_list)), np.nan, np.float32)

    for yi, sigma in enumerate(noise_list):
        for xi, p in enumerate(p_list):
            finals = []
            for s in seeds:
                out = train_swimmer_ppo(
                    episodes=episodes, T=T, d=d,
                    ws_k=ws_k, ws_p=float(p),
                    obs_noise_std=float(sigma),
                    eval_every=eval_every,
                    seed=int(s)
                )
                finals.append(final_eval_mean(out["history"]["eval_rand"]))
            heat[yi, xi] = float(np.mean(finals))
            print(f"[p={p:.3f}, σ={sigma:.3f}] → final(mean-last-10% eval) = {heat[yi, xi]:.2f} over {len(seeds)} seeds")

    return heat

def plot_heatmap(heat, p_list, noise_list, title="Final performance (late-window mean)"):
    plt.figure(figsize=(7.2, 5.4))
    im = plt.imshow(heat, origin='lower', aspect='auto',
                    extent=[min(p_list), max(p_list), min(noise_list), max(noise_list)])
    cbar = plt.colorbar(im)
    cbar.set_label("Eval return (mean of last 10%)")
    plt.xlabel("WS rewiring probability p")
    plt.ylabel("Observation noise σ")
    plt.title(title)
    # optional: ticks at provided grid values
    plt.xticks(p_list)
    plt.yticks(noise_list)
    plt.tight_layout()
    plt.savefig("grid_heatmap.png", dpi=150)
    print("Saved heatmap to grid_heatmap.png")

if __name__ == "__main__":
    # sweep settings (tweak as needed)
    p_list = np.round(np.linspace(0.0, 1, 10), 3)        # x-axis (SW % via p)
    noise_list = np.round(np.linspace(0.0, 0.5, 5), 3)    # y-axis (observation noise std)

    heat = sweep_heatmap(
        p_list=p_list,
        noise_list=noise_list,
        seeds=(0,1),          # ↑ to (0,1,2,3,4) for paper-quality CIs
        episodes=600,         # ↑ for stronger convergence
        T=4096,               # ↑ for larger batches
        ws_k=20,              # even, < d
        d=400,                # reservoir size
        eval_every=10
    )
    plot_heatmap(heat, p_list, noise_list,
                 title="Swimmer-v4: PPO on WS-Reservoir — Final Eval (mean last 10%)")
