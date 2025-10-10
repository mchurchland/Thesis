# reservoir_diagnostics_heatmaps.py
# ------------------------------------------------------------
# Compute 4 task-agnostic diagnostics for reservoir computers:
#   MC  = linear Memory Capacity (sum R^2 over delays)
#   IPC = Information Processing Capacity (Legendre-based nonlinear memory; sum over orders/delays)
#   KR  = Kernel Rank / Quality (effective rank of state covariance)
#   GR  = Generalization Rank (effective rank of state-difference under small input noise)
#
# Adds per-run instrumentation for debugging:
#   - Saturation fraction (sat_frac)
#   - Effective ranks of states and preactivations (erank_X, erank_pre)
#   - Spectral radius vs spectral norm and non-normality proxy (rho, smax, nn)
#   - Linearized controllability proxy rank (rank_k)
#   - State-scale stats across units (std stats; frac near-zero)
#
# The script builds reservoirs under different *features* (connectivity, Dale's law,
# spectral norm policy) and renders heatmaps for each diagnostic where rows/cols
# enumerate feature settings.
#
# GPU acceleration: uses PyTorch; will use CUDA if available (falls back to CPU).
#
# NOTE: To use a real C. elegans connectome, provide:
#   - CE_ADJ_PATH: path to an NxN numpy .npy array of weights (float; signed allowed).
#   - CE_EI_PATH  (optional): path to length-N numpy .npy array with values in {-1,0,+1}
#       (+1=excitatory, -1=inhibitory, 0=unknown/mixed).
# If CE_ADJ_PATH is None, a synthetic WS graph is used for the "cel" option.
#
# Usage:
#   python reservoir_diagnostics_heatmaps.py
#
# Outputs:
#   heatmap_MC.png, heatmap_IPC.png, heatmap_KR.png, heatmap_GR.png
# ------------------------------------------------------------

import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import Tensor

# -------- external helpers (expected in your repo) ----------
from util import load_connectome, imsave_heatmap  # keep your versions
from stats import compute_IPC, compute_KR, compute_GR, compute_MC  # keep your versions

# =================== Config: paths & defaults ===================

CE_ADJ_PATH = "Connectome/ce_adj.npy"
CE_EI_PATH  = "Connectome/ce_ei.npy"

N_DEFAULT = 400

# Time-series lengths for diagnostics
WASHOUT = 1000
T_TRAIN = 10000
T_TEST  = 2000
SEED    = 0

# Ridge regularization for linear readouts
RIDGE_ALPHA = 1e-4

# WS topology defaults
WS_K = 40  # must be even and < N

# Probe and dynamics
INPUT_SCALE = 0.1
LEAK = 1.0

# GR noise
PERTURB_STD = 0.01

# IPC / MC settings
IPC_MAX_DELAY = 50
IPC_MAX_ORDER = 3
MC_MAX_DELAY  = 300

# Spectral gain target (via spectral norm)
SR_TARGET_OPT = 0.99

# Instrumentation thresholds / settings
SAT_THRESH = 2.0          # |pre| > SAT_THRESH counts as saturated
NEAR_ZERO_STD = 1e-3      # unit considered "inactive" if std < this
K_CONTROLLABILITY = 100   # horizon for controllability proxy

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# =================== Utilities ===================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





def degree_matched_shuffle_directed(A: np.ndarray, tries: int = 10_000,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Directed degree-preserving randomization by double-edge swaps on 0/1 adjacency."""
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
    """Power iteration estimate of spectral radius on DEVICE."""
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


def scale_to_sr(W: Tensor, mode: str, sr_target: float | None = None) -> Tensor:
    """
    Control amplification via spectral norm (non-normal-safe).
    mode: 'natural' or 'target'
    """
    if mode == "natural":
        # Reverted: do NOT change/scale at all.
        return W
    assert sr_target is not None and sr_target > 0
    smax = torch.linalg.svdvals(W)[0]
    if float(smax) < 1e-9:
        return W
    return (sr_target / smax) * W


def apply_dales_law(W: Tensor, dale_mode: str, ei_labels: Tensor | None,
                    ei_ratio: float = 0.8, rng: np.random.Generator | None = None) -> Tensor:
    """
    Enforce outgoing sign constraints per neuron by zeroing disallowed signs.
    dale_mode: 'none', 'e80i20', 'ei_from_cel'
    """
    if dale_mode == "none":
        return W
    N = W.shape[0]
    if dale_mode == "ei_from_cel" and ei_labels is not None:
        signs = ei_labels.to(W.device)  # in {-1,0,+1}
    else:
        if rng is None:
            rng = np.random.default_rng(SEED)
        n_exc = int(round(ei_ratio * N))
        signs_np = np.ones(N, dtype=np.float32)
        signs_np[n_exc:] = -1.0
        rng.shuffle(signs_np)
        signs = torch.from_numpy(signs_np).to(W.device)

    W_signed = W.clone()
    s = signs.view(-1, 1)
    mask_exc = (s > 0)
    mask_inh = (s < 0)
    W_signed[mask_exc & (W_signed < 0)] = 0.0
    W_signed[mask_inh & (W_signed > 0)] = 0.0
    W_signed.fill_diagonal_(0.0)
    return W_signed


# =================== Reservoir builder ===================

def build_reservoir(
    feature_conn: str,         # 'cel', 'deg_shuffle', 'ws_p=1.0', 'ws_p=0.1', 'ws_p=0.0'
    feature_weights: str,      # 'bio', 'rand_disc', 'rand_gauss'
    feature_dale: str,         # 'none', 'e80i20', 'ei_from_cel'
    feature_sr: str,           # 'natural', 'target'
    N: int,
    ce_W_bio: np.ndarray | None,
    ce_ei: np.ndarray | None,
    ws_k: int,
    sr_target: float,
    input_scale: float,
    seed: int
) -> tuple[Tensor, Tensor, Tensor | None, float, float]:
    """
    Returns:
      W: [N,N] torch tensor on DEVICE
      Win: [N,1] torch tensor on DEVICE
      EI: optional torch tensor (+1/0/-1) if available/used
      smax_nat: spectral norm before scaling
      smax_post: spectral norm after scaling
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    if feature_conn == "cel":
        if ce_W_bio is None:
            # synthetic CE-like: WS mask with signed Gaussian weights
            A = ws_adjacency(N, ws_k, 0.1, rng).astype(np.float32)
            W = (A != 0).astype(np.float32) * rng.normal(0.0, 1.0, size=(N, N)).astype(np.float32)
            ce_ei_t = None
        else:
            W = ce_W_bio.copy().astype(np.float32)
            N = W.shape[0]
            # per-row L1 normalization to temper hub gain
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

    # Torch
    Wt = torch.from_numpy(W).to(DEVICE)

    # Dale
    ei_t = torch.from_numpy(ce_ei).to(DEVICE) if (feature_dale == "ei_from_cel" and ce_ei is not None) else None
    Wt = apply_dales_law(Wt, feature_dale, ei_t, ei_ratio=0.8, rng=rng)

    # Spectral norms
    smax_nat = spectral_norm(Wt)
    Wt = scale_to_sr(Wt, "target" if feature_sr == "target" else "natural",
                     sr_target=(SR_TARGET_OPT if feature_sr == "target" else None))
    smax_post = spectral_norm(Wt)

    # Input weights
    Win = torch.randn(Wt.shape[0], 1, device=DEVICE) * INPUT_SCALE

    return Wt, Win, ei_t, smax_nat, smax_post


# =================== Local simulator with preactivation capture ===================

@torch.no_grad()
def run_reservoir_with_pre(W: Tensor, Win: Tensor, u: Tensor, leak: float) -> tuple[Tensor, Tensor]:
    """
    u: [T, 1] input sequence on DEVICE
    Returns:
      X:   [T, N] states
      Pre: [T, N] pre-activations (W @ z + Win @ u_t)
    """
    N = W.shape[0]
    T = u.shape[0]
    z = torch.zeros(N, device=DEVICE)
    X = torch.zeros(T, N, device=DEVICE)
    Pre = torch.zeros(T, N, device=DEVICE)
    for t in range(T):
        pre = W @ z + (Win @ u[t:t+1, :].T).squeeze()
        h = torch.tanh(pre)
        z = (1 - leak) * z + leak * h
        X[t] = z
        Pre[t] = pre
    return X, Pre


# =================== Instrumentation helpers ===================

@torch.no_grad()
def effective_rank(X: Tensor) -> float:
    """
    Effective rank via entropy of singular value proportions.
    X is [T, D]. Center first.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    s = torch.clamp(s, min=1e-12)
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    return float(torch.exp(H))


@torch.no_grad()
def controllability_erank(W: Tensor, Win: Tensor, leak: float, Ddiag_mean: Tensor, K: int) -> float:
    """
    Linearized dynamic: A = (1-leak)I + leak*Dbar@W, where Dbar = diag(1 - tanh(pre)^2) averaged over train.
    Build controllability matrix C = [A^0 Win, A^1 Win, ..., A^K Win] and return effective rank(C).
    """
    N = W.shape[0]
    I = torch.eye(N, device=W.device)
    A = (1 - leak) * I + leak * (Ddiag_mean @ W)
    cols = []
    v = Win  # [N,1]
    for _ in range(K + 1):
        cols.append(v)
        v = A @ v
    C = torch.cat(cols, dim=1)  # [N, K+1]
    # use SVD of C directly (no centering; it's not timeseries)
    s = torch.linalg.svdvals(C)
    s = torch.clamp(s, min=1e-12)
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    return float(torch.exp(H))


# =================== Diagnostics driver (with instrumentation) ===================

def run_diagnostics_for_reservoir(W: Tensor, Win: Tensor) -> dict:
    """Run reservoir on clean/noisy probes and compute diagnostics + instrumentation."""
    T_total = WASHOUT + T_TRAIN + T_TEST

    # Zero-mean probe in [-1,1]
    u = (torch.rand(T_total, 1, device=DEVICE) * 2.0 - 1.0)
    u = u - u.mean()

    X, Pre = run_reservoir_with_pre(W, Win, u, LEAK)
    Xn, _  = run_reservoir_with_pre(W, Win, u + PERTURB_STD * torch.randn_like(u), LEAK)

    # Split
    Xtr = X[WASHOUT:WASHOUT+T_TRAIN]
    Xte = X[WASHOUT+T_TRAIN:]
    Pre_tr = Pre[WASHOUT:WASHOUT+T_TRAIN]
    utr = u[WASHOUT:WASHOUT+T_TRAIN]
    ute = u[WASHOUT+T_TRAIN:]

    # Core diagnostics
    MC_total, _ = compute_MC(Xtr, Xte, utr, ute, MC_MAX_DELAY, RIDGE_ALPHA)
    IPC_total   = compute_IPC(Xtr, Xte, utr, ute, IPC_MAX_DELAY, IPC_MAX_ORDER, RIDGE_ALPHA)
    KR_val      = compute_KR(Xtr)
    GR_val      = compute_GR(Xtr, Xn[WASHOUT:WASHOUT+T_TRAIN])

    # Instrumentation
    # 1) Saturation fraction
    sat_frac = float((Pre_tr.abs() > SAT_THRESH).float().mean())

    # 2) Effective dimensionalities
    erank_X   = effective_rank(Xtr)
    erank_pre = effective_rank(Pre_tr)

    # 3) Radius vs norm
    rho  = spectral_radius_power(W)
    smax = spectral_norm(W)
    nn   = smax - rho

    # 4) Linearized controllability proxy
    #    Dbar = mean diag(1 - tanh(pre)^2) over train
    Dbar_diag = (1.0 - torch.tanh(Pre_tr)**2).mean(dim=0)            # [N]
    Dbar = torch.diag(Dbar_diag)                                     # [N,N]
    rank_k = controllability_erank(W, Win, LEAK, Dbar, K_CONTROLLABILITY)

    # 5) State scale
    std_per_unit = Xtr.std(dim=0)                                    # [N]
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


# =================== Main benchmark loop ===================

def main():
    set_seed(SEED)
    ce_W_bio, ce_ei = load_connectome(CE_ADJ_PATH, CE_EI_PATH)

    # If CE provided, set N to its size, else use default
    N = ce_W_bio.shape[0] if ce_W_bio is not None else N_DEFAULT
    if WS_K >= N:
        raise ValueError(f"WS_K must be < N. Got WS_K={WS_K}, N={N}")

    # Rows = connectivity/weights variants
    row_defs = [
        ("CEL + bioW",             dict(conn="cel",         weights="bio")),
        ("CEL + randN",            dict(conn="cel",         weights="rand_gauss")),
        ("CEL-shuf + randN",       dict(conn="deg_shuffle", weights="rand_gauss")),
        ("WS p=1.0 + randN",       dict(conn="ws_p=1.0",    weights="rand_gauss")),
        ("WS p=0.1 + randN",       dict(conn="ws_p=0.1",    weights="rand_gauss")),
        ("WS p=0.0 + randN",       dict(conn="ws_p=0.0",    weights="rand_gauss")),
        ("WS p=0.1 + randDisc",    dict(conn="ws_p=0.1",    weights="rand_disc")),
    ]

    # Cols = Dale’s law / spectral norm variants
    col_defs = [
        ("no-Dale, SR=natural",    dict(dale="none",       sr="natural")),
        ("no-Dale, SR=0.99",       dict(dale="none",       sr="target")),
        ("Dale 80:20, SR=0.99",    dict(dale="e80i20",     sr="target")),
        ("Dale from CEL, SR=0.99", dict(dale="ei_from_cel",sr="target")),
        ("Dale 80:20, SR=natural", dict(dale="e80i20",     sr="natural")),
    ]

    # Initialize arrays
    H_MC  = np.full((len(row_defs), len(col_defs)), np.nan, np.float32)
    H_IPC = np.full((len(row_defs), len(col_defs)), np.nan, np.float32)
    H_KR  = np.full((len(row_defs), len(col_defs)), np.nan, np.float32)
    H_GR  = np.full((len(row_defs), len(col_defs)), np.nan, np.float32)

    for (ri, (rname, rconf)), (ci, (cname, cconf)) in itertools.product(
            enumerate(row_defs), enumerate(col_defs)):

        build_hdr = f"\n=== Building reservoir: ROW '{rname}' × COL '{cname}'"
        try:
            Wt, Win, _, smax_nat, smax_post = build_reservoir(
                feature_conn=rconf["conn"],
                feature_weights=rconf["weights"],
                feature_dale=cconf["dale"],
                feature_sr=cconf["sr"],
                N=N,
                ce_W_bio=ce_W_bio,
                ce_ei=ce_ei,
                ws_k=WS_K,
                sr_target=SR_TARGET_OPT,
                input_scale=INPUT_SCALE,
                seed=SEED + ri*37 + ci*101
            )
            if cconf["sr"] == "natural":
                build_hdr += f" (||W||₂_nat={smax_nat:.3f} → unchanged)"
            else:
                build_hdr += f" (||W||₂_nat={smax_nat:.3f} → scaled to {smax_post:.3f})"
            print(build_hdr + " ===")

            scores = run_diagnostics_for_reservoir(Wt, Win)

            # Fill heatmaps
            H_MC[ri, ci]  = scores["MC"]
            H_IPC[ri, ci] = scores["IPC"]
            H_KR[ri, ci]  = scores["KR"]
            H_GR[ri, ci]  = scores["GR"]

            # Print core + instrumentation succinctly
            if cconf["sr"] == "natural":
                tail = f"||W||₂_nat={smax_nat:.3f} → unchanged"
            else:
                tail = f"||W||₂_nat={smax_nat:.3f} → {smax_post:.3f}"

            print(
                "MC={MC:.2f} | IPC={IPC:.2f} | KR={KR:.2f} | GR={GR:.2f} | "
                "sat={sat:.3f} | erank_X={erx:.1f} | erank_pre={erp:.1f} | "
                "rho={rho:.3f} | smax={smax:.3f} | nn={nn:.3f} | "
                "rank_k={rk:.1f} | frac_std≈0={fz:.3f} | "
                "std[mean/med/min/max]={sm:.4f}/{sd:.4f}/{smin:.4f}/{smaxu:.4f} | "
                "{tail}".format(
                    MC=scores["MC"], IPC=scores["IPC"], KR=scores["KR"], GR=scores["GR"],
                    sat=scores["sat_frac"], erx=scores["erank_X"], erp=scores["erank_pre"],
                    rho=scores["rho"], smax=scores["smax"], nn=scores["nn"],
                    rk=scores["rank_k"], fz=scores["frac_near_zero"],
                    sm=scores["std_mean"], sd=scores["std_med"],
                    smin=scores["std_min"], smaxu=scores["std_max"],
                    tail=tail
                )
            )

        except Exception as e:
            print(build_hdr + " ===")
            print(f"  [SKIP] Configuration failed: {e}")

    # Save heatmaps
    row_labels = [r[0] for r in row_defs]
    col_labels = [c[0] for c in col_defs]

    imsave_heatmap(H_MC,  row_labels, col_labels, "Linear Memory Capacity (MC)",                   "heatmap_MC.png")
    imsave_heatmap(H_IPC, row_labels, col_labels, "Information Processing Capacity (IPC, L=1..3)", "heatmap_IPC.png")
    imsave_heatmap(H_KR,  row_labels, col_labels, "Kernel Rank / Quality (effective rank)",        "heatmap_KR.png")
    imsave_heatmap(H_GR,  row_labels, col_labels, "Generalization Rank (effective rank of Δstate)", "heatmap_GR.png")

    print("\nDone. Heatmaps saved: heatmap_MC.png, heatmap_IPC.png, heatmap_KR.png, heatmap_GR.png")


if __name__ == "__main__":
    main()
