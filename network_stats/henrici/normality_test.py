#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Henrici departure from normality and a Kreiss-constant lower bound
for the C. elegans connectome.

Uses:
  util.load_connectome("Connectome/ce_adj.npy", "Connectome/ce_ei.npy")

Outputs (printed):
  - spectral radius rho(W)
  - Henrici departure delta_H(W)
  - Frobenius commutator norm ||W W^T - W^T W||_F  (sanity check)
  - Kreiss lower bound K_lb(W) based on a grid search
"""

import numpy as np
import argparse

from numpy.linalg import eigvals, norm, svd

from util.util import load_connectome


def henrici_departure(W: np.ndarray) -> float:
    """
    Henrici departure from normality (Henrici's measure):

      delta_H(W) = sqrt( ||W||_F^2 - sum_i |lambda_i|^2 )

    = 0 iff W is normal. Larger => more non-normal.
    """
    # ensure float/complex
    W = np.asarray(W, dtype=np.complex128)
    fro2 = norm(W, "fro") ** 2
    lam = eigvals(W)
    lam2_sum = np.sum(np.abs(lam) ** 2)
    delta = float(np.sqrt(max(fro2 - lam2_sum, 0.0)))
    return delta


def commutator_fro(W: np.ndarray) -> float:
    """
    ||W W^* - W^* W||_F, another zero-iff-normal diagnostic.
    Not Henrici's measure, but useful to sanity check.
    """
    W = np.asarray(W, dtype=np.complex128)
    C = W @ W.conj().T - W.conj().T @ W
    return float(norm(C, "fro"))


def kreiss_lower_bound(
    W: np.ndarray,
    n_radius: int = 40,
    n_theta: int = 64,
    radius_factor: float = 1.5,
    eps: float = 1e-4,
) -> float:
    """
    Crude LOWER BOUND on Kreiss constant:

      K(W) = sup_{|z| > rho(W)} (|z| - rho(W)) * ||(zI - W)^{-1}||_2

    We approximate this by a grid over radii and angles.
    This is expensive but fine for CE-size matrices.

    Parameters:
      n_radius:   # of radial samples between rho+eps and radius_factor*rho
      n_theta:    # of angular samples on [0, 2pi)
      radius_factor: outer radius as multiple of rho
      eps:        small offset above rho

    Returns:
      K_lb: float, lower bound on K(W)
    """
    W = np.asarray(W, dtype=np.complex128)
    n = W.shape[0]

    # spectral radius
    lam = eigvals(W)
    rho = float(np.max(np.abs(lam)))

    if rho <= 0:
        return 0.0

    r_min = rho + eps
    r_max = max(r_min * 1.0001, radius_factor * rho)  # ensure > r_min
    radii = np.linspace(r_min, r_max, n_radius)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    K_lb = 0.0
    I = np.eye(n, dtype=np.complex128)

    for r in radii:
        for th in thetas:
            z = r * np.exp(1j * th)
            # (zI - W)^{-1}
            try:
                A = z * I - W
                # spectral norm of inverse via largest singular value
                # svd returns singular values sorted desc by default
                svals = svd(A, compute_uv=False)
                s_min = float(svals[-1])
                if s_min <= 0:
                    continue
                norm_inv = 1.0 / s_min
            except np.linalg.LinAlgError:
                continue

            val = (r - rho) * norm_inv
            if val > K_lb:
                K_lb = val

    return float(K_lb)


def parse_args():
    ap = argparse.ArgumentParser(description="Compute Henrici and Kreiss lower bound for CE connectome.")
    ap.add_argument("--ce-adj", default="Connectome/ce_adj.npy")
    ap.add_argument("--ce-ei",  default="Connectome/ce_ei.npy")
    return ap.parse_args()


def main():
    args = parse_args()

    W_bio, ce_ei, _ = load_connectome(args.ce_adj, args.ce_ei)
    if W_bio is None:
        raise FileNotFoundError(f"Could not load connectome at {args.ce_adj}")

    # ensure square, remove self-loops for safety (load_connectome should already)
    if W_bio.shape[0] != W_bio.shape[1]:
        raise ValueError("Connectome adjacency must be square.")
    np.fill_diagonal(W_bio, 0.0)

    # compute metrics
    lam = eigvals(W_bio.astype(np.complex128))
    rho = float(np.max(np.abs(lam)))
    delta_H = henrici_departure(W_bio)
    comm_F = commutator_fro(W_bio)
    K_lb = kreiss_lower_bound(W_bio)

    print(f"rho(W)                      = {rho:.6e}")
    print(f"Henrici departure delta_H   = {delta_H:.6e}")
    print(f"||W W* - W* W||_F (check)   = {comm_F:.6e}")
    print(f"Kreiss constant lower bound = {K_lb:.6e}")


if __name__ == "__main__":
    main()
