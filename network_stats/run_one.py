import torch
from torch import Tensor

# ---- repo helpers (reuse your utils/stats) ----
from network_stats.stats import compute_IPC, compute_KR, compute_GR, compute_MC
@torch.no_grad()
def run_reservoir_with_pre(W: Tensor, Win: Tensor, u: Tensor, leak: float) -> tuple[Tensor, Tensor]:
    """
    Basic ESN-style update loop with pre-activation tracking.
    See: https://github.com/cknd/pyESN/blob/master/pyESN.py for a similar state update.
    """
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


def run_one(W: Tensor, Win: Tensor, leak: float, device: torch.device,WASHOUT: int,
            PERTURB_STD: float, T_TRAIN: int, T_TEST: int,
            MC_MAX_DELAY: int, IPC_MAX_DELAY: int, IPC_ORDERS: list[int],
            RIDGE_ALPHA: float) -> dict:
    """
    End-to-end reservoir evaluation computing MC/IPC/KR/GR and controllability metrics.
    Wraps the ESN update plus metrics pipeline; see reservoirpy/pyESN for similar evaluation flows:
    [1] Jaeger, H. (2001). Short term memory in echo state networks.    """
    T_total = WASHOUT + T_TRAIN + T_TEST
    u = (torch.rand(T_total, 1, device=device) * 2.0 - 1.0) ## rescale to [-1, 1]

    X, _ = run_reservoir_with_pre(W, Win, u, leak)
    Xn, _  = run_reservoir_with_pre(W, Win, u + PERTURB_STD * torch.randn_like(u), leak)

    Xtr = X[WASHOUT:WASHOUT+T_TRAIN] ## t_train
    Xte = X[WASHOUT+T_TRAIN:] ## t_test
    utr = u[WASHOUT:WASHOUT+T_TRAIN] ## u_train
    ute = u[WASHOUT+T_TRAIN:] ## u_test

    MC_total, _ = compute_MC(Xtr, Xte, utr, ute, MC_MAX_DELAY, RIDGE_ALPHA,device)
    IPC_total   = compute_IPC(Xtr, Xte, utr, ute, IPC_MAX_DELAY, RIDGE_ALPHA,device, IPC_ORDERS)
    KR_val      = compute_KR(Xtr)
    GR_val      = compute_GR(Xtr, Xn[WASHOUT:WASHOUT+T_TRAIN])
    return dict(
        MC=MC_total, IPC=IPC_total, KR=KR_val, GR=GR_val,
    )
