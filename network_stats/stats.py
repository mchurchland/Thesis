from network_stats.stats_util import corr2_score,legendre_P,effective_rank_from_states,ridge_fit_predict
from torch import Tensor
import torch ; import numpy as np

def compute_IPC(
    Xtr: Tensor,
    Xte: Tensor,
    utr: Tensor,
    ute: Tensor,
    max_delay: int,
    alpha: float,
    device: torch.device,
    orders: list[int] = [1, 3, 5],
    capacity_threshold: float = 0.0,
) -> float:
    """
    Information Processing Capacity (approx.): sum of for Legendre targets
    P_k(u_{t - d}) for k in orders, d=1..max_delay.

    This version batches all Legendre orders for a fixed delay d, so that
    we do ONE Cholesky factorization per delay and solve for all targets
    in one go.
    Terms with corr^2 below capacity_threshold are omitted.

    See: Dambre et al., 2012, Sci. Rep. 2:514; batching pattern similar to scikit-learn Ridge
    multi-target solves (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py).
    """
    total = 0.0

    for d in range(1, max_delay + 1):
        # shared slices for this delay
        base_tr = utr[:-d]          # [T_train - d, 1] or [T_train - d]
        base_te = ute[:-d]          # [T_test  - d, 1] or [T_test  - d]
        Xtr_d   = Xtr[d:]             # [T_train - d, N]
        Xte_d   = Xte[d:]             # [T_test  - d, N]

        # build all Legendre targets for orders k=1..max_order
        ytr_cols = []
        yte_cols = []
        for k in orders:
            ytr_k = legendre_P(base_tr, k)
            yte_k = legendre_P(base_te, k)

            # ensure 2D [T, 1]
            if ytr_k.dim() == 1:
                ytr_k = ytr_k.unsqueeze(1)
            if yte_k.dim() == 1:
                yte_k = yte_k.unsqueeze(1)

            ytr_cols.append(ytr_k)
            yte_cols.append(yte_k)

        # [T_train - d, max_order], [T_test - d, max_order]
        Ytr = torch.cat(ytr_cols, dim=1).to(device)
        Yte = torch.cat(yte_cols, dim=1).to(device)

        # ONE ridge solve for all orders k at this delay
        Yhat = ridge_fit_predict(Xtr_d, Ytr, Xte_d, alpha, DEVICE=device)
        # Yhat: [T_test - d, max_order]

        for j in range(Ytr.shape[1]):
            score = corr2_score(Yte[:, j], Yhat[:, j])
            if score >= capacity_threshold:
                total += score

    return float(total)


def compute_KR(X: Tensor) -> float:
    ## this is just the rank of the matrix, which is the effective rank
    return effective_rank_from_states(X)


def compute_MC(Xtr: Tensor, Xte: Tensor, utr: Tensor, ute: Tensor, max_delay: int, alpha: float,device:torch.device) -> tuple[float, np.ndarray]:
    """
    Linear memory capacity using Jaeger's squared-correlation definition.
    Inputs/targets are assumed zero-mean.
    See: Jaeger, 2002, GMD Report 152; reservoirpy metric analogue:
         https://github.com/reservoirpy/reservoirpy/blob/master/reservoirpy/metrics/memory_capacity.py
    """
    capacities = []
    for tau in range(1, max_delay + 1):
        ytr = utr[:-tau]
        yte = ute[:-tau]
        Xtr_d = Xtr[tau:] ##so that they are the same dimensiobn
        Xte_d = Xte[tau:] ##so that they are the same dimesions
        yhat = ridge_fit_predict(Xtr_d, ytr, Xte_d, alpha,DEVICE=device)
        capacities.append(corr2_score(yte, yhat))
    capacities = np.array(capacities, dtype=np.float32)
    return float(np.sum(capacities)), capacities


def compute_GR(X_clean: Tensor, X_noisy: Tensor) -> float:
    """
    Generalization rank: effective rank of state difference across small perturbations.
    Lower => more robust/generalizable.
    Related to robustness metrics using effective rank; see Roy & Vetterli, 2007, IEEE SPL 14:649-652.
    """
    D = X_noisy - X_clean ##this is nice I like this, here is whats going on, we add noise
    # to the input of the reservoir, then we subtract that from the version without noise
    # and then we compute the effective rank, essentially we see how many dimensions the noise adds
    # between the clean and noisy states.
    return effective_rank_from_states(D)
