from torch import Tensor
import torch

def legendre_P(x: Tensor, order: int) -> Tensor:
    # x in [-1,1]
    if order == 1:
        return x
    elif order == 2:
        return 0.5 * (3 * x**2 - 1)
    elif order == 3:
        return 0.5 * (5 * x**3 - 3 * x)
    else:
        raise ValueError("Supported orders: 1..3")

def effective_rank_from_states(X: Tensor) -> float:
    """
    Kernel rank via effective rank of centered states.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    s = torch.clamp(s, min=1e-12)
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    erank = torch.exp(H)
    return float(erank)


def ridge_fit_predict(Xtr: Tensor, ytr: Tensor, Xte: Tensor, alpha: float) -> Tensor:
    """
    Ridge readout with feature standardization and pinv fallback.
    """
    from heatmap import DEVICE
    # center and scale columns (train stats)
    mu = Xtr.mean(dim=0, keepdim=True)
    sig = Xtr.std(dim=0, keepdim=True) + 1e-8
    Xtr_n = (Xtr - mu) / sig
    Xte_n = (Xte - mu) / sig

    XT_X = Xtr_n.T @ Xtr_n
    D = XT_X.shape[0]
    A = XT_X + alpha * torch.eye(D, device=DEVICE)
    b = Xtr_n.T @ ytr
    try:
        w = torch.linalg.solve(A, b)
    except RuntimeError:
        w = torch.linalg.pinv(A) @ b
    return Xte_n @ w

def r2_score(y_true: Tensor, y_pred: Tensor) -> float:
    y_true_c = y_true - y_true.mean()
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum(y_true_c**2) + 1e-12
    return float(1.0 - (ss_res / ss_tot))

@torch.no_grad()
def effective_rank(X: Tensor) -> float:
    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc)
    s = torch.clamp(s, min=1e-12)
    p = s / torch.sum(s)
    H = -torch.sum(p * torch.log(p))
    return float(torch.exp(H))