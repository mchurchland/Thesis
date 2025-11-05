import torch
from torch import Tensor

# ---- repo helpers (reuse your utils/stats) ----
from util.util import load_connectome, build_reservoir
from network_stats.stats import compute_IPC, compute_KR, compute_GR, compute_MC
from util.util import degree_matched_shuffle_directed
from network_stats.stats_util import  effective_rank
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

def run_one(W: Tensor, Win: Tensor, leak: float, device: torch.device,WASHOUT: int,
            PERTURB_STD: float, T_TRAIN: int, T_TEST: int,
            MC_MAX_DELAY: int, IPC_MAX_DELAY: int, IPC_MAX_ORDER: int,
            RIDGE_ALPHA: float, K_CONTROLLABILITY: int,
            SAT_THRESH: float, NEAR_ZERO_STD: float) -> dict:
    T_total = WASHOUT + T_TRAIN + T_TEST
    u = (torch.rand(T_total, 1, device=device) * 2.0 - 1.0)
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

    Dbar_diag = (1.0 - torch.tanh(Pre_tr)**2).mean(dim=0)
    Dbar = torch.diag(Dbar_diag)
    rank_k = controllability_erank(W, Win, leak, Dbar, K_CONTROLLABILITY)
    std_per_unit = Xtr.std(dim=0)

    return dict(
        MC=MC_total, IPC=IPC_total, KR=KR_val, GR=GR_val,
        sat_frac=float((Pre_tr.abs() > SAT_THRESH).float().mean()),
        erank_X=effective_rank(Xtr),
        erank_pre=effective_rank(Pre_tr),
        rank_k=rank_k,
        frac_near_zero=float((std_per_unit < NEAR_ZERO_STD).float().mean())
    )