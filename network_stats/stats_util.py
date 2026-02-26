from torch import Tensor
import torch

def legendre_P(x: Tensor, order: int) -> Tensor:
    """
    Low-order Legendre polynomials P_n on [-1, 1] (n=1..3) via torch.special.legendre_polynomial_p.
    """
    if order > 3 or order < 0:
        raise ValueError("Supported orders: 1..3")
    return torch.special.legendre_polynomial_p(x, order)


def effective_rank_from_states(X: Tensor) -> float:
    """
    Shannon effective rank of centered states (Roy & Vetterli, 2007, IEEE Signal Processing Letters 14:649-652).
    Uses torch.linalg.svdvals; see https://github.com/pytorch/pytorch/blob/main/torch/linalg/__init__.py
    """
    Xc = X - X.mean(dim=0, keepdim=True) ## normalize the vec dim 0 is time normalize with respect to time
    s = torch.linalg.svdvals(Xc) ## this just gives sqrt(x^2) of eigen values, which is abs(eigen)
    s = torch.clamp(s, min=1e-12) ## the iegen values are positive, and hence we just need to make sure that they are not zero or some relaly small value
    p = s / torch.sum(s) ## mean is 0 stdev of 1 WHY AM I SUMMING S this is not normalizing btu seing what part of the distribution each individual s makes
    H = -torch.sum(p * torch.log(p)) #shannon entropy expected amount of information needed to encode the distribution
    #if h is high lots of variance across many dimensions, if variance constrained to a few dimensions, then h is low
    erank = torch.exp(H)
    ##this gives us the number of the dimensions that are needed to encode the state of the system
    return float(erank)


def ridge_fit_predict(
    Xtr: Tensor,
    ytr: Tensor,
    Xte: Tensor,
    alpha: float,
    DEVICE: torch.device,
    eps: float = 1e-6,
) -> Tensor:
    """
    Ridge regression:

        w = (X^T X + alpha I)^(-1) X^T y

    Supports:
        ytr: [T] or [T, K] (multiple targets).

    Uses ONE solve with multiple RHS instead of one per target.
    See: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py
    """
    Xtr = Xtr.to(DEVICE)
    Xte = Xte.to(DEVICE)
    ytr = ytr.to(DEVICE)

    single_target = False
    if ytr.dim() == 1:
        ytr = ytr.unsqueeze(1)   # [T, 1]
        single_target = True

    T_train, n_feat = Xtr.shape
    Xt = Xtr.transpose(0, 1)     # [N, T]

    # Gram matrix
    G = Xt @ Xtr                 # [N, N]
    G = G + (alpha + eps) * torch.eye(
        n_feat, device=DEVICE, dtype=Xtr.dtype
    )

    # All right-hand sides at once
    B = Xt @ ytr                 # [N, K]

    # Single linear solve for all targets
    W = torch.linalg.solve(G, B) # [N, K]

    # Predictions
    yhat = Xte @ W               # [T_test, K]

    if single_target:
        return yhat.squeeze(1)

    return yhat


def r2_score(y_true: Tensor, y_pred: Tensor) -> float:
    """Coefficient of determination R^2 (see sklearn.metrics.r2_score: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_regression.py#L104)."""
    y_true_c = y_true - y_true.mean()
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum(y_true_c**2) + 1e-12
    return float(1.0 - (ss_res / ss_tot))
