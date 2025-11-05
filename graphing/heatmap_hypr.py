# reservoir_diagnostics_heatmaps.py
# ------------------------------------------------------------
# Hyperparameter sweep heatmaps (columns):
#   srtarget ∈ {0.6, 0.8, 0.95, 1.05}   <-- spectral radius targets
#   LEAK     ∈ {0.6, 0.8, 1.0}
#   INPUT    ∈ {0.1, 0.5, 1.0, 1.5}
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
#   sat_frac, erank_X, erank_pre, rho (nat/post), nn=smax-rho,
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

# -------- existing helpers (already in your repo) ----------
from util.util import (
    load_connectome,
    imsave_heatmap,
    spectral_radius_power,
    spectral_norm,
    build_reservoir,
    set_seed      # must support target_sr + nnz_target + drive_idx
)
from stats import compute_IPC, compute_KR, compute_GR, compute_MC  # keep your versions


# =================== Paths & defaults ===================

CE_ADJ_PATH = "Connectome/ce_adj.npy"   # NxN float32, signed allowed
CE_EI_PATH  = "Connectome/ce_ei.npy"    # length-N in {-1,0,+1}; optional

N_DEFAULT = 400

# Time-series lengths
WASHOUT = 1000
T_TRAIN = 10000
T_TEST  = 2000
SEED    = 0

# Readout ridge
RIDGE_ALPHA = 1e-4

# WS topology default
WS_K = 40  # even, < N

# ER topology default (we will match nnz to CE, so p is only a starting point)
ER_P = 0.1

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




# ---------- edge-count matching (keep nnz equal across models) ----------

def _count_nnz(A: np.ndarray) -> int:
    B = (np.abs(A) > 0)
    np.fill_diagonal(B, False)
    return int(B.sum())


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


# =================== Single run (produces MC, IPC, KR, GR + stats) ===================

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


# =================== Main: sweep (ρ-scaling + equal nnz) ===================

def main():
    set_seed(SEED)
    # Your util may return (W, EI, name2idx) or just (W, EI)
    out = load_connectome(CE_ADJ_PATH, CE_EI_PATH)
    if len(out) == 3:
        ce_W_bio, ce_ei, ce_name2idx = out
    else:
        ce_W_bio, ce_ei = out
        ce_name2idx = None

    N = ce_W_bio.shape[0] if ce_W_bio is not None else N_DEFAULT
    if WS_K >= N:
        raise ValueError(f"WS_K must be < N. Got WS_K={WS_K}, N={N}")

    nnz_target = None
    if ce_W_bio is not None:
        nnz_target = _count_nnz(ce_W_bio)
        print(f"[INFO] Matching nnz across models to CE: {nnz_target} edges")

    # Build sensory drive indices for C. elegans if names are available
    cel_drive_idx = None
    if isinstance(ce_name2idx, dict):
        touch_names = ("FLPR","FLPL","ASHL","ASHR","IL1VL","IL1VR","OLQDL","OLQDR","OLQVR","OLQVL")
        food_names  = ("ADFL","ADFR","ASGR","ASGL","ASIL","ASIR","ASJR","ASJL")
        sel = []
        for n in list(touch_names) + list(food_names):
            if n in ce_name2idx:
                sel.append(ce_name2idx[n])
        if len(sel) > 0:
            cel_drive_idx = np.array(sorted(set(sel)), dtype=np.int32)
            print(f"[INFO] Driving CE sensory neurons (count={len(cel_drive_idx)}).")
        else:
            print("[WARN] Names file present but none of the requested sensory names matched.")
    elif ce_W_bio is not None:
        print("[WARN] No CE names file found — CEL rows will use random Win.")

    # Rows (feature variants) — include standard ESN (directed ER)
    row_defs = [
        ("CEL + bioW",                 dict(conn="cel",         weights="bio",        use_cel_drive=True)),
        ("CEL + randN",                dict(conn="cel",         weights="rand_gauss", use_cel_drive=True)),
        ("CEL-shuf + randN",           dict(conn="deg_shuffle", weights="rand_gauss", use_cel_drive=True)),
        ("WS p=1.0 + randN",           dict(conn="ws_p=1.0",    weights="rand_gauss", use_cel_drive=False)),
        ("WS p=0.1 + randN",           dict(conn="ws_p=0.1",    weights="rand_gauss", use_cel_drive=False)),
        ("WS p=0.0 + randN",           dict(conn="ws_p=0.0",    weights="rand_gauss", use_cel_drive=False)),
        ("WS p=0.1 + randDisc",        dict(conn="ws_p=0.1",    weights="rand_disc",  use_cel_drive=False)),
        ("Std ESN ER p=0.1 + randN",   dict(conn=f"er_p={ER_P}", weights="rand_gauss", use_cel_drive=False)),
        ("Std ESN ER p=0.1 + randDisc",dict(conn=f"er_p={ER_P}", weights="rand_disc",  use_cel_drive=False)),
    ]

    # X-axis sweep (4 * 3 * 4 = 48 columns) — srtarget is spectral radius
    SWEEP_SR    = [0.6, 0.8, 0.95, 1.05]
    SWEEP_LEAK  = [0.6, 0.8, 1.0]
    SWEEP_U     = [0.1, 0.5, 1.0, 1.5]
    col_params = [(s, l, u) for s in SWEEP_SR for l in SWEEP_LEAK for u in SWEEP_U]
    col_labels = [f"ρ{sm:.2f}|L{lk:.1f}|U{ui:.2f}" for (sm, lk, ui) in col_params]

    # Allocate heatmaps
    H_MC  = np.full((len(row_defs), len(col_params)), np.nan, np.float32)
    H_IPC = np.full((len(row_defs), len(col_params)), np.nan, np.float32)
    H_KR  = np.full((len(row_defs), len(col_params)), np.nan, np.float32)
    H_GR  = np.full((len(row_defs), len(col_params)), np.nan, np.float32)

    # Fixed Dale for this sweep (change if you want)
    DALE_MODE = "none"

    for ri, (rname, rconf) in enumerate(row_defs):
        for ci, (target_sr, leak, in_scale) in enumerate(col_params):
            hdr = f"\n=== ROW '{rname}' × COL ρ*={target_sr:.2f}, leak={leak:.1f}, U={in_scale:.2f} ==="
            try:
                drive_idx = cel_drive_idx if (rconf.get("use_cel_drive", False) and ce_W_bio is not None) else None
                Wt, Win, _, rho_nat, rho_post = build_reservoir(
                    feature_conn=rconf["conn"],
                    feature_weights=rconf["weights"],
                    feature_dale=DALE_MODE,
                    target_sr=target_sr,                 # <--- spectral radius scaling
                    N=N,
                    ce_W_bio=ce_W_bio,
                    ce_ei=ce_ei,
                    ws_k=WS_K,
                    input_scale=in_scale,
                    seed=SEED + ri*37 + ci*101,
                    drive_idx=drive_idx,
                    nnz_target=_count_nnz(ce_W_bio) if (ce_W_bio is not None) else None
                )
                print(hdr + f" (ρ_nat={rho_nat:.3f} → ρ_post={rho_post:.3f})")
                scores = run_one(Wt, Win, leak)

                H_MC [ri, ci] = scores["MC"]
                H_IPC[ri, ci] = scores["IPC"]
                H_KR [ri, ci] = scores["KR"]
                H_GR [ri, ci] = scores["GR"]

                print(
                    "MC={MC:.2f} | IPC={IPC:.2f} | KR={KR:.2f} | GR={GR:.2f} | "
                    "sat={sat:.3f} | erank_X={erx:.1f} | erank_pre={erp:.1f} | "
                    "ρ={rho:.3f} | ||W||₂={smax:.3f} | nn={nn:.3f} | "
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
    imsave_heatmap(H_MC,  row_labels, col_labels, "MC vs (ρ*, leak, input)", "heatmap_MC.png")
    imsave_heatmap(H_IPC, row_labels, col_labels, "IPC vs (ρ*, leak, input)", "heatmap_IPC.png")
    imsave_heatmap(H_KR,  row_labels, col_labels, "KR vs (ρ*, leak, input)", "heatmap_KR.png")
    imsave_heatmap(H_GR,  row_labels, col_labels, "GR vs (ρ*, leak, input)", "heatmap_GR.png")

    # ========== NEW: MC vs GR scatter (one color per row) ==========
    plt.figure(figsize=(12, 6), dpi=150)
    cmap = plt.get_cmap("tab10")
    for ri, rname in enumerate(row_labels):
        mc_row = H_MC[ri]
        gr_row = H_GR[ri]
        m = ~np.isnan(mc_row) & ~np.isnan(gr_row)
        if not np.any(m):
            continue
        plt.scatter(gr_row[m], mc_row[m], s=28, alpha=0.8, label=rname, color=cmap(ri % 10))

    plt.xlabel("GR (effective rank of Δstate)")
    plt.ylabel("MC (linear memory capacity)")
    plt.title("MC vs GR (each point = one column config; colors = rows/topologies)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("scatter_MC_vs_GR.png")
    plt.close()

    print("\nDone. Heatmaps saved: heatmap_MC.png, heatmap_IPC.png, heatmap_KR.png, heatmap_GR.png")
    print("Also saved: scatter_MC_vs_GR.png")


if __name__ == "__main__":
    main()
