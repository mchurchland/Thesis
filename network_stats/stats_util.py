from torch import Tensor
import torch

def legendre_P(x: Tensor, order: int) -> Tensor:
    """
    Legendre polynomials P_n on [-1, 1] via torch.special.legendre_polynomial_p.
    """
    if order < 0:
        raise ValueError("Supported orders: n >= 0")
    return torch.special.legendre_polynomial_p(x, order)


def effective_rank_from_states(X: Tensor) -> float:
    """
    Shannon effective rank of centered states (Roy & Vetterli, 2007, IEEE Signal Processing Letters 14:649-652).
    Uses torch.linalg.svdvals; see https://github.com/pytorch/pytorch/blob/main/torch/linalg/__init__.py
    """
    Xc = X - X.mean(dim=0, keepdim=True) ## centers the vec dim 0 is time -> center with respect to time
    s = torch.linalg.svdvals(Xc) ## this gives the amount of scaling in each direction, these values are equal to AA^T = u\sigmav^tvsigmau^t = u\sqrt(\sigma)u^t
    ## Hence here the vals afdsare equal to, |eigen of s_l|
    ## some more intuition AA^T is the covariance matrix of our neurons, if neurons fire together or have simular patterns of high charge then their entires, 
    # AA^T_{ij} and AA^T_{ji} will be large,
    # if these neurons are low charge for the duration of time, or if they have orthognal patterns of activation then these values will be low.
    # If nothing changes over time: Every z_t is the same. the matrix has rank 1
    # If states move in many independent directions over time: Then X^T X accumulates variance in many orthogonal directions.
    #this matrix will have high rank
    s = torch.clamp(s, min=1e-12) ## the iegen values are semi positive positive definite , and hence we just need to make sure that they are not zero or some really small value
    p = s / torch.sum(s) ## divide by sum, if one dominates it will be close to one, these should some to 1 and probabilities,
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



def corr2_score(y_true: Tensor, y_pred: Tensor, eps: float = 1e-12) -> float:
    """
    Squared Pearson correlation used in Jaeger-style memory capacity.
    """
    y_true_c = y_true - y_true.mean() ## remove the means
    y_pred_c = y_pred - y_pred.mean() ## remove the means 
    num = torch.sum(y_true_c * y_pred_c) ## this is top on pearson correlation
    den = torch.sqrt(torch.sum(y_true_c**2) * torch.sum(y_pred_c**2)) + eps
    corr2 = (num / den) ** 2
    return float(torch.clamp(corr2, 0.0, 1.0))