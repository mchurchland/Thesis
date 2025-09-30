# reservoir_diagnostics_heatmaps.py
# ------------------------------------------------------------
# Hyperparameter sweep heatmaps (columns):
#   smax_target ∈ {0.6, 0.8, 0.95, 1.05}
#   LEAK        ∈ {0.6, 0.8, 1.0}
#   INPUT_SCALE ∈ {0.1, 0.5, 1.0, 1.5}
#
# Rows: connectivity/weight variants (CEL, degree-shuffle, WS, and a
#       standard ESN topology (directed Erdős–Rényi, "ER")).
#
# Diagnostics:
#   MC  = linear Memory Capacity
#   IPC = Information Processing Capacity (Legendre 1..3)
#   KR  = Kernel Rank (effective rank of state covariance)
#   GR  = Generalization Rank (effective rank of Δstate)
#
# Instrumentation (printed per config):
#   sat_frac, erank_X, erank_pre, rho, smax, nn=smax-rho,
#   controllability erank (rank_k), and state std stats
#
# GPU: uses PyTorch; runs on CUDA if available.
# ------------------------------------------------------------

import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import Tensor

# =================== Paths & defaults ===================

# Optional C. elegans connectome paths
CE_ADJ_PATH = "Connectome/ce_adj.npy"   # NxN float32, signed allowed
CE_EI_PATH  = "Connectome/ce_ei.npy"    # length-N in {-1,0,+1}; optional

# Fallback size if no CE provided
N_DEFAULT = 400

# Time-series lengths
WASHOUT = 1000
T_TRAIN = 10000
T_TEST  = 2000
SEED    = 0

# Readout ridge
RIDGE_ALPHA = 1e-4

# WS topology default (used for synthetic)
WS_K = 40  # even, < N

# ER topology default
ER_P = 0.1  # edge probability for standard ESN

# GR noise
PERTURB_STD = 0.01

# IPC / MC settings
IPC_MAX_DELAY = 50
IPC_MAX_ORDER = 3
MC_MAX_DELAY  = 300

# Instrument thresholds
SAT_THRESH = 2.0
NEAR_ZERO_STD = 1e-3
K_CONTROLLABILITY = 100

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# =================== Utilities ===================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_connectome(adj_path: str | None, ei_path: str | None):
    W_bio, ei_labels = None, None
    if adj_path is not None and os.path.isfile(adj_path):
        W_bio = np.load(adj_path).astype(np.float32, copy=False)
        assert W_bio.ndim == 2 and W_bio.shape[0] == W_bio.shape[1], "Adjacency must be square."
    if ei_path is not None and os.path.isfile(ei_path):
        ei_labels = np.load(ei_path).astype(np.float32, copy=False)
        if W_bio is not None:
            assert ei_labels.ndim == 1 and ei_labels.shape[0] == W_bio.shape[0], "EI length mismatch."
    return W_bio, ei_labels

def imsave_heatmap(data: np.ndarray, row_labels, col_labels, title: str, fname: str):
    plt.figure(figsize=(1.6 + 0.25*len(col_labels), 1.6 + 0.9*len(row_labels)))
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    im = plt.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(im)
    plt.xticks(range(len(col_labels)), col_labels, rotation=90, fontsize=7)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()
    print(f"Saved {fname}")

def ws_adjacency(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    assert k % 2 == 0 and k < n and 0.0 <= p <= 1.0
    A = np.zeros((n, n), dtype=bool)
    # regular ring
    for i in range(n):
        for j in range(1, k // 2 + 1):
            A[i, (i + j) % n] = True
            A[i, (i - j) % n] = True
    # rewire symmetrically
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                old = (i + j) % n
                if not A[i, old]:
                    continue
                A[i, old] = False; A[old, i] = False
                candidates = np.where(~A[i] & (np.arange(n) != i))[0]
                if candidates.size == 0:
                    A[i, old] = True; A[old, i] = True
                else:
                    new = rng.choice(candidates)
                    A[i, new] = True; A[new, i] = True
    np.fill_diagonal(A, False)
    return A.astype(np.float32)

def er_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Directed Erdős–Rényi adjacency with edge prob p (independent directions)."""
    A = rng.random((n, n)) < p
    np.fill_diagonal(A, False)
    return A.astype(np.float32)

def degree_matched_shuffle_directed(A: np.ndarray, tries: int = 10_000,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    A = A.copy().astype(bool)
    n = A.shape[0]
    np.fill_diagonal(A, False)
    edges = np.argwhere(A)
    m = edges.shape[0]
    if m < 2:
        return A.astype(np.float32)
    for _ in range(tries):
        idx = rng.choice(m, size=2, replace=False)
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]
        if len({a, b, c, d}) < 4:
            continue
        if a == d or c == b:
            continue
        if A[a, d] or A[c, b]:
            continue
        A[a, b] = False; A[c, d] = False
        A[a, d] = True;  A[c, b] = True
        edges[idx[0]] = [a, d]
        edges[idx[1]] = [c, b]
    return A.astype(np.float32)

def spectral_norm(W: Tensor) -> float:
    return float(torch.linalg.svdvals(W)[0])

@torch.no_grad()
def spectral_radius_power(W: Tensor, iters: int = 200) -> float:
    n = W.shape[0]
    v = torch.randn(n, device=W.device)
    v = v / (v.norm() + 1e-12)
    lam = 0.0
    for _ in range(iters):
        v = W @ v
        nrm = v.norm()
        if float(nrm) < 1e-12:
            break
        v = v / nrm
        lam = float((v @ (W @ v)) / (v @ v + 1e-12))
    return abs(lam)

def scale_to_smax(W: Tensor, target_smax: float | None) -> Tensor:
    """Scale W so that ||W||_2 = target_smax; if None, return unchanged."""
    if target_smax is None:
        return W
    smax = torch.linalg.svdvals(W)[0]
    if float(smax) < 1e-9:
        return W
    return (target_smax / smax) * W

def apply_dales_law(W: Tensor, dale_mode: str, ei_labels: Tensor | None,
                    ei_ratio: float = 0.8, seed: int = 0) -> Tensor:
    if dale_mode == "none":
        return W
    N = W.shape[0]
    if dale_mode == "ei_from_cel" and ei_labels is not None:
        signs = ei_labels.to(W.device)  # {-1,0,+1}
    else:
        rng = np.random.default_rng(seed)
        n_exc = int(round(ei_ratio * N))
        signs_np = np.ones(N, dtype=np.float32)
        signs_np[n_exc:] = -1.0
        rng.shuffle(signs_np)
        signs = torch.from_numpy(signs_np).to(W.device)
    W2 = W.clone()
    s = signs.view(-1, 1)
    mask_exc = (s > 0)
    mask_inh = (s < 0)
    W2[mask_exc & (W2 < 0)] = 0.0
    W2[mask_inh & (W2 > 0)] = 0.0
    W2.fill_diagonal_(0.0)
    return W2


# =================== Readout & scores ===================

def ridge_fit_predict(Xtr: Tensor, ytr: Tensor, Xte: Tensor, alpha: float) -> Tensor:
    XT_X = Xtr.T @ Xtr
    D = XT_X.shape[0]
    reg = alpha * torch.eye(D, device=XT_X.device)
    A = XT_X + reg
    b = Xtr.T @ ytr
    w = torch.linalg.solve(A, b)
    return Xte @ w

def r2_score(y_true: Tensor, y_pred: Tensor) -> float:
    y_true_c = y_true - y_true.mean()
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum(y_true_c**2) + 1e-12
    return float(1.0 - (ss_res / ss_tot))

def compute_MC(Xtr: Tensor, Xte: Tensor, utr: Tensor, ute: Tensor, max_delay: int, alpha: float):
    r2s = []
    for tau in range(1, max_delay + 1):
        ytr = utr[:-tau]
        yte = ute[:-tau]
        Xtr_d = Xtr[tau:]
        Xte_d = Xte[tau:]
        yhat = ridge_fit_predict(Xtr_d, ytr, Xte_d, alpha)
        r2s.append(r2_score(yte, yhat))
    r2s = np.array(r2s, dtype=np.float32)
    return float(np.sum(np.clip(r2s, 0.0, 1.0))), r2s

def legendre_P(x: Tensor, order: int) -> Tensor:
    if order == 1:
        return x
    if order == 2:
        return 0.5 * (3 * x**2 - 1)
    if order == 3:
        return 0.5 * (5 * x**3 - 3 * x)
    raise ValueError("Supported orders: 1..3")

def compute_IPC(Xtr: Tensor, Xte: Tensor, utr: Tensor, ute: Tensor,
                max_delay: int, max_order: int, alpha: float) -> float:
    def scale01_to_m11(u: Tensor) -> Tensor:
        u0 = (u - u.min()) / (u.max() - u.min() + 1e-12)
        return u0 * 2 - 1
    utr_s = scale01_to_m11(utr)
    ute_s = scale01_to_m11(ute)
    total = 0.0
    for d in range(1, max_delay + 1):
        for k in range(1, max_order + 1):
            ytr = legendre_P(utr_s[:-d], k)
            yte = legendre_P(ute_s[:-d], k)
            Xtr_d = Xtr[d:]
            Xte_d = Xte[d:]
            yhat = ridge_fit_predict(Xtr_d, ytr, Xte_d, alpha)
            total += max(0.0, r2_score(yte, yhat))
    return float(total)

def effective_rank_from_states(X: Tensor) -> float:
    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    s = torch.clamp(s, min=1e-12)
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    return float(torch.exp(H))

def compute_KR(X: Tensor) -> float:
    return effective_rank_from_states(X)

def compute_GR(X_clean: Tensor, X_noisy: Tensor) -> float:
    D = X_noisy - X_clean
    return effective_rank_from_states(D)


# =================== Reservoir dynamics ===================

@torch.no_grad()
def run_reservoir_with_pre(W: Tensor, Win: Tensor, u: Tensor, leak: float) -> tuple[Tensor, Tensor]:
    N = W.shape[0]
    T = u.shape[0]
    z = torch.zeros(N, device=W.device)
    X = torch.zeros(T, N, device=W.device)
    Pre = torch.zeros(T, N, device=W.device)
    for t in range(T):
        pre = W @ z + (Win @ u[t:t+1, :].T).squeeze()
        h = torch.tanh(pre)
        z = (1 - leak) * z + leak * h
        X[t] = z
        Pre[t] = pre
    return X, Pre


# =================== Instrumentation ===================

@torch.no_grad()
def effective_rank(X: Tensor) -> float:
    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    s = torch.clamp(s, min=1e-12)
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    return float(torch.exp(H))

@torch.no_grad()
def controllability_erank(W: Tensor, Win: Tensor, leak: float, Ddiag_mean: Tensor, K: int) -> float:
    N = W.shape[0]
    I = torch.eye(N, device=W.device)
    A = (1 - leak) * I + leak * (Ddiag_mean @ W)
    cols = []
    v = Win
    for _ in range(K + 1):
        cols.append(v)
        v = A @ v
    C = torch.cat(cols, dim=1)  # [N, K+1]
    s = torch.linalg.svdvals(C)
    s = torch.clamp(s, min=1e-12)
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    return float(torch.exp(H))


# =================== Builder ===================

def build_reservoir(
    feature_conn: str,         # 'cel', 'deg_shuffle', 'ws_p=1.0', 'ws_p=0.1', 'ws_p=0.0', 'er_p=0.1'
    feature_weights: str,      # 'bio', 'rand_disc', 'rand_gauss'
    feature_dale: str,         # 'none', 'e80i20', 'ei_from_cel'
    target_smax: float | None, # scale to this spectral norm (None = unchanged)
    N: int,
    ce_W_bio: np.ndarray | None,
    ce_ei: np.ndarray | None,
    ws_k: int,
    input_scale: float,
    seed: int
) -> tuple[Tensor, Tensor, Tensor | None, float, float]:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    if feature_conn == "cel":
        if ce_W_bio is None:
            A = ws_adjacency(N, ws_k, 0.1, rng).astype(np.float32)
            W = (A != 0).astype(np.float32) * rng.normal(0.0, 1.0, size=(N, N)).astype(np.float32)
            ce_ei_t = None
        else:
            W = ce_W_bio.copy().astype(np.float32)
            N = W.shape[0]
            row_abs = np.sum(np.abs(W), axis=1, keepdims=True) + 1e-8
            W = W / row_abs
            ce_ei_t = torch.from_numpy(ce_ei) if ce_ei is not None else None

    elif feature_conn == "deg_shuffle":
        if ce_W_bio is None:
            raise ValueError("Degree-matched shuffle requires CE adjacency.")
        A = (ce_W_bio != 0).astype(np.float32)
        As = degree_matched_shuffle_directed(A, tries=20_000, rng=rng)
        if feature_weights == "bio":
            vals = ce_W_bio[ce_W_bio != 0].astype(np.float32)
            rng.shuffle(vals)
            W = np.zeros_like(As, dtype=np.float32)
            W[As != 0] = vals[: np.count_nonzero(As)]
        else:
            W = As.copy().astype(np.float32)
        row_abs = np.sum(np.abs(W), axis=1, keepdims=True) + 1e-8
        W = W / row_abs
        ce_ei_t = torch.from_numpy(ce_ei) if ce_ei is not None else None
        N = W.shape[0]

    elif feature_conn.startswith("ws_p="):
        p = float(feature_conn.split("=")[1])
        A = ws_adjacency(N, ws_k, p, rng).astype(np.float32)
        W = (A != 0).astype(np.float32)
        ce_ei_t = None

    elif feature_conn.startswith("er_p="):
        p = float(feature_conn.split("=")[1])
        A = er_adjacency(N, p, rng).astype(np.float32)
        W = (A != 0).astype(np.float32)  # start with 0/1 mask; weights below
        ce_ei_t = None

    else:
        raise ValueError(f"Unknown feature_conn: {feature_conn}")

    # Weight scheme
    mask = (np.abs(W) > 0).astype(np.float32)
    if feature_weights == "rand_disc":
        vals = rng.choice([-1.0, 1.0], size=mask.shape).astype(np.float32)
        W = mask * vals
    elif feature_weights == "rand_gauss":
        vals = rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
        W = mask * vals
    elif feature_weights == "bio":
        if ce_W_bio is None:
            # emulate heavy-tail signed weights; then row-normalize
            m = int(mask.sum())
            mags  = rng.lognormal(mean=-1.0, sigma=0.5, size=m).astype(np.float32)
            signs = rng.choice([-1.0, 1.0], size=m).astype(np.float32)
            vals  = mags * signs
            W_new = np.zeros_like(W, dtype=np.float32)
            W_new[mask != 0] = vals
            row_abs = np.sum(np.abs(W_new), axis=1, keepdims=True) + 1e-8
            W = W_new / row_abs
        # else CE weights already normalized

    # Torchify
    Wt = torch.from_numpy(W).to(DEVICE)

    # Dale (optional; fixed to 'none' in main sweep unless changed)
    if feature_dale != "none":
        ei_t = torch.from_numpy(ce_ei).to(DEVICE) if (feature_dale == "ei_from_cel" and ce_ei is not None) else None
        Wt = apply_dales_law(Wt, feature_dale, ei_t, seed=seed)
    else:
        ei_t = None

    smax_nat = spectral_norm(Wt)
    Wt = scale_to_smax(Wt, target_smax)
    smax_post = spectral_norm(Wt)

    Win = torch.randn(Wt.shape[0], 1, device=DEVICE) * input_scale

    return Wt, Win, ei_t, smax_nat, smax_post


# =================== Diagnostics driver ===================

def run_one(W: Tensor, Win: Tensor, leak: float):
    T_total = WASHOUT + T_TRAIN + T_TEST
    u = (torch.rand(T_total, 1, device=DEVICE) * 2.0 - 1.0)
    u = u - u.mean()

    X, Pre = run_reservoir_with_pre(W, Win, u, leak)
    Xn, _  = run_reservoir_with_pre(W, Win, u + PERTURB_STD * torch.randn_like(u), leak)

    Xtr = X[WASHOUT:WASHOUT+T_TRAIN]
    Xte = X[WASHOUT+T_TRAIN:]
    Pre_tr = Pre[WASHOUT:WASHOUT+T_TRAIN]
    utr = u[WASHOUT:WASHOUT+T_TRAIN]
    ute = u[WASHOUT+T_TRAIN:]

    MC_total, _ = compute_MC(Xtr, Xte, utr, ute, MC_MAX_DELAY, RIDGE_ALPHA)
    IPC_total   = compute_IPC(Xtr, Xte, utr, ute, IPC_MAX_DELAY, IPC_MAX_ORDER, RIDGE_ALPHA)
    KR_val      = compute_KR(Xtr)
    GR_val      = compute_GR(Xtr, Xn[WASHOUT:WASHOUT+T_TRAIN])

    sat_frac = float((Pre_tr.abs() > SAT_THRESH).float().mean())
    erank_X   = effective_rank(Xtr)
    erank_pre = effective_rank(Pre_tr)
    rho  = spectral_radius_power(W)
    smax = spectral_norm(W)
    nn   = smax - rho
    Dbar_diag = (1.0 - torch.tanh(Pre_tr)**2).mean(dim=0)
    Dbar = torch.diag(Dbar_diag)
    rank_k = controllability_erank(W, Win, leak, Dbar, K_CONTROLLABILITY)
    std_per_unit = Xtr.std(dim=0)
    frac_near_zero = float((std_per_unit < NEAR_ZERO_STD).float().mean())
    std_mean = float(std_per_unit.mean())
    std_med  = float(std_per_unit.median())
    std_max  = float(std_per_unit.max())
    std_min  = float(std_per_unit.min())

    return dict(
        MC=MC_total, IPC=IPC_total, KR=KR_val, GR=GR_val,
        sat_frac=sat_frac, erank_X=erank_X, erank_pre=erank_pre,
        rho=rho, smax=smax, nn=nn, rank_k=rank_k,
        frac_near_zero=frac_near_zero, std_mean=std_mean, std_med=std_med,
        std_max=std_max, std_min=std_min
    )


# =================== Main: sweep on X-axis ===================

def main():
    set_seed(SEED)
    ce_W_bio, ce_ei = load_connectome(CE_ADJ_PATH, CE_EI_PATH)

    N = ce_W_bio.shape[0] if ce_W_bio is not None else N_DEFAULT
    if WS_K >= N:
        raise ValueError(f"WS_K must be < N. Got WS_K={WS_K}, N={N}")

    # Rows (feature variants) — added standard ESN (directed ER) rows
    row_defs = [
        ("CEL + bioW",                 dict(conn="cel",         weights="bio")),
        ("CEL + randN",                dict(conn="cel",         weights="rand_gauss")),
        ("CEL-shuf + randN",           dict(conn="deg_shuffle", weights="rand_gauss")),
        ("WS p=1.0 + randN",           dict(conn="ws_p=1.0",    weights="rand_gauss")),
        ("WS p=0.1 + randN",           dict(conn="ws_p=0.1",    weights="rand_gauss")),
        ("WS p=0.0 + randN",           dict(conn="ws_p=0.0",    weights="rand_gauss")),
        ("WS p=0.1 + randDisc",        dict(conn="ws_p=0.1",    weights="rand_disc")),
        ("Std ESN ER p=0.1 + randN",   dict(conn=f"er_p={ER_P}", weights="rand_gauss")),
        ("Std ESN ER p=0.1 + randDisc",dict(conn=f"er_p={ER_P}", weights="rand_disc")),
    ]

    # X-axis sweep (4 * 3 * 4 = 48 columns)
    SWEEP_SMAX  = [0.6, 0.8, 0.95, 1.05]
    SWEEP_LEAK  = [0.6, 0.8, 1.0]
    SWEEP_U     = [0.1, 0.5, 1.0, 1.5]
    col_params = [(s, l, u) for s in SWEEP_SMAX for l in SWEEP_LEAK for u in SWEEP_U]
    col_labels = [f"s{sm:.2f}|L{lk:.1f}|U{ui:.2f}" for (sm, lk, ui) in col_params]

    # Allocate heatmaps
    H_MC  = np.full((len(row_defs), len(col_params)), np.nan, np.float32)
    H_IPC = np.full((len(row_defs), len(col_params)), np.nan, np.float32)
    H_KR  = np.full((len(row_defs), len(col_params)), np.nan, np.float32)
    H_GR  = np.full((len(row_defs), len(col_params)), np.nan, np.float32)

    # Fixed Dale for this sweep (change to 'e80i20' or 'ei_from_cel' to compare)
    DALE_MODE = "none"

    for ri, (rname, rconf) in enumerate(row_defs):
        for ci, (target_smax, leak, in_scale) in enumerate(col_params):
            hdr = f"\n=== ROW '{rname}' × COL smax={target_smax:.2f}, leak={leak:.1f}, U={in_scale:.2f} ==="
            try:
                Wt, Win, _, smax_nat, smax_post = build_reservoir(
                    feature_conn=rconf["conn"],
                    feature_weights=rconf["weights"],
                    feature_dale=DALE_MODE,
                    target_smax=target_smax,
                    N=N,
                    ce_W_bio=ce_W_bio,
                    ce_ei=ce_ei,
                    ws_k=WS_K,
                    input_scale=in_scale,
                    seed=SEED + ri*37 + ci*101
                )
                print(hdr + f" (||W||₂_nat={smax_nat:.3f} → {smax_post:.3f})")
                scores = run_one(Wt, Win, leak)

                H_MC [ri, ci] = scores["MC"]
                H_IPC[ri, ci] = scores["IPC"]
                H_KR [ri, ci] = scores["KR"]
                H_GR [ri, ci] = scores["GR"]

                print(
                    "MC={MC:.2f} | IPC={IPC:.2f} | KR={KR:.2f} | GR={GR:.2f} | "
                    "sat={sat:.3f} | erank_X={erx:.1f} | erank_pre={erp:.1f} | "
                    "rho={rho:.3f} | smax={smax:.3f} | nn={nn:.3f} | "
                    "rank_k={rk:.1f} | frac_std≈0={fz:.3f} | "
                    "std[mean/med/min/max]={sm:.4f}/{sd:.4f}/{smin:.4f}/{smaxu:.4f}".format(
                        MC=scores["MC"], IPC=scores["IPC"], KR=scores["KR"], GR=scores["GR"],
                        sat=scores["sat_frac"], erx=scores["erank_X"], erp=scores["erank_pre"],
                        rho=scores["rho"], smax=scores["smax"], nn=scores["nn"],
                        rk=scores["rank_k"], fz=scores["frac_near_zero"],
                        sm=scores["std_mean"], sd=scores["std_med"],
                        smin=scores["std_min"], smaxu=scores["std_max"]
                    )
                )
            except Exception as e:
                print(hdr + f"\n  [SKIP] Configuration failed: {e}")

    row_labels = [r[0] for r in row_defs]
    imsave_heatmap(H_MC,  row_labels, col_labels, "MC vs (smax, leak, input)", "heatmap_MC.png")
    imsave_heatmap(H_IPC, row_labels, col_labels, "IPC vs (smax, leak, input)", "heatmap_IPC.png")
    imsave_heatmap(H_KR,  row_labels, col_labels, "KR vs (smax, leak, input)", "heatmap_KR.png")
    imsave_heatmap(H_GR,  row_labels, col_labels, "GR vs (smax, leak, input)", "heatmap_GR.png")
    print("\nDone. Heatmaps saved: heatmap_MC.png, heatmap_IPC.png, heatmap_KR.png, heatmap_GR.png")


if __name__ == "__main__":
    main()
