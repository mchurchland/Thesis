from stats_util import legendre_P,effective_rank_from_states,ridge_fit_predict,r2_score
from torch import Tensor
import torch ; import numpy as np

def compute_IPC(Xtr: Tensor, Xte: Tensor, utr: Tensor, ute: Tensor,
                max_delay: int, max_order: int, alpha: float) -> float:
    """
    Information Processing Capacity (approx.): sum of R^2 for Legendre targets
    P_k(u_{t - d}) for k=1..max_order, d=1..max_delay.
    Inputs are rescaled to [-1,1] per-split.
    """ 
    # rescale u to [-1,1]
    def to_m11(u: Tensor) -> Tensor:
        umax = u.max()
        umin = u.min()
        return (2.0 * (u - umin) / (umax - umin + 1e-12)) - 1.0

    utr_s = to_m11(utr)
    ute_s = to_m11(ute)

    total = 0.0
    for d in range(1, max_delay + 1):
        for k in range(1, max_order + 1):
            ytr = legendre_P(utr_s[:-d], k) ##train take off d amount of input
            yte = legendre_P(ute_s[:-d], k) ##test take off d amount of input
            Xtr_d = Xtr[d:] ## 
            Xte_d = Xte[d:]
            yhat = ridge_fit_predict(Xtr_d, ytr, Xte_d, alpha)
            total += max(0.0, r2_score(yte, yhat))
    return float(total)


def compute_KR(X: Tensor) -> float:
    ## this is just the rank of the matrix, which is the effective rank
    return effective_rank_from_states(X)


def compute_MC(Xtr: Tensor, Xte: Tensor, utr: Tensor, ute: Tensor, max_delay: int, alpha: float) -> tuple[float, np.ndarray]:
    """
    Linear memory capacity (sum of R^2 over delays).
    Inputs/targets are assumed zero-mean. just linear version of ipc
    """
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


def compute_GR(X_clean: Tensor, X_noisy: Tensor) -> float:
    """
    Generalization rank: effective rank of state difference across small perturbations.
    Lower ⇒ more robust/generalizable.
    """
    D = X_noisy - X_clean ##this is nice I like this, here is whats going on, we add noise
    # to the input of the reservoir, then we subtract that from the version without noise
    # and then we compute the effective rank, essentially we see how many dimensions the noise adds
    # between the clean and noisy states.
    return effective_rank_from_states(D)