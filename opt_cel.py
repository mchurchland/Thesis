#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize C. elegans connectome reservoir weights for MC/IPC/KR/GR
using EA-NOMAD (EANOMAD PyPI).

Decision variables:
    x_i in [-1, 1] for each nonzero synapse (excluding diagonal)
    W_new[nz_i] = W_bio[nz_i] + x_i * delta_scale

Objective:
    Weighted scalarization of mean(MC, IPC, KR, GR) across col_params,
    normalized vs baseline (original connectome).
    Optional L2 penalty to prevent large drift.

EANOMAD API:
    opt = EANOMAD("EA" or "rEA", population_size, dimension, objective_fn, ...)
    best_x, best_fit = opt.run(generations=G)
:contentReference[oaicite:0]{index=0}
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path
import torch

from util.util import load_connectome, build_reservoir
from network_stats.run_one import run_one
from EANOMAD import EANOMAD  # pip install EANOMAD :contentReference[oaicite:1]{index=1}
WASHOUT        = 1000
T_TRAIN        = 10000
T_TEST         = 2000
RIDGE_ALPHA    = 1e-4
IPC_MAX_DELAY  = 50
IPC_MAX_ORDER  = 3
MC_MAX_DELAY   = 300
PERTURB_STD    = 0.01
SAT_THRESH     = 2.0
NEAR_ZERO_STD  = 1e-3
K_CONTROLLABILITY = 100

# ------------------------- helpers -------------------------

def save_csv(path: str, header: list[str], rows: list[tuple]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run_scores_for_matrix(
    WS_K: int,
    W_bio_mat: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    device: torch.device,
    seed_base: int = 0,
) -> dict[str, np.ndarray]:
    """Return arrays over col_params for MC/IPC/KR/GR."""
    scores = {k: [] for k in ("MC", "IPC", "KR", "GR")}
    for ci, (target_sr, leak, in_scale) in enumerate(col_params):
        try:
            Wt, Win, _, _, _ = build_reservoir(
                feature_conn="cel",
                feature_weights="bio",
                feature_dale="none",
                target_sr=target_sr,
                N=W_bio_mat.shape[0],
                ce_W_bio=W_bio_mat,
                ce_ei=ce_ei,
                ws_k=WS_K,
                input_scale=in_scale,
                seed=seed_base + ci * 101,
                drive_idx=None,
                nnz_target=None,
                DEVICE=device
            )
            Wt = Wt.to(device)
            Win = Win.to(device)
            sc = run_one(Wt, Win, leak, device,WASHOUT,PERTURB_STD,T_TRAIN,T_TEST,MC_MAX_DELAY,IPC_MAX_DELAY,IPC_MAX_ORDER,RIDGE_ALPHA,\
                         K_CONTROLLABILITY,SAT_THRESH,NEAR_ZERO_STD)
        except Exception as e:
            print(e)


        for k in scores:
            scores[k].append(float(sc[k]))

    return {k: np.asarray(v, dtype=np.float32) for k, v in scores.items()}


def make_col_params(
    rho_targets: list[float],
    leaks: list[float],
    input_scales: list[float],
) -> list[tuple[float, float, float]]:
    return [(r, l, u) for r in rho_targets for l in leaks for u in input_scales]


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-adj", type=str, required=True,
                    help="Path prefix for load_connectome (same as your other scripts).")
    ap.add_argument("--ce-ei", type=str, required=True,
                    help="Path prefix for load_connectome (same as your other scripts).")
    ap.add_argument("--out-dir", type=str, default="eanomad_opt")
    ap.add_argument("--device", type=str, default="cuda")

    # reservoir construction sweep during evaluation
    ap.add_argument("--rho-targets", type=float, nargs="+", default=[0.9, 1.0, 1.1])
    ap.add_argument("--leaks", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    ap.add_argument("--input-scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])

    # optimization hyperparams
    ap.add_argument("--mode", type=str, choices=["EA", "rEA"], default="rEA",
                    help="EA runs NOMAD on random slices each gen; rEA only on mutated coords. :contentReference[oaicite:2]{index=2}")
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--population-size", type=int, default=32)
    ap.add_argument("--subset-size", type=int, default=20,
                    help="Coords refined per NOMAD call, keep <=49. :contentReference[oaicite:3]{index=3}")
    ap.add_argument("--bounds", type=float, default=0.1,
                    help="Half-width of NOMAD box around slice. :contentReference[oaicite:4]{index=4}")
    ap.add_argument("--max-bb-eval", type=int, default=150,
                    help="NOMAD evaluations per call. :contentReference[oaicite:5]{index=5}")
    ap.add_argument("--n-mutate-coords", type=int, default=5,
                    help="Coords reset per mutation. :contentReference[oaicite:6]{index=6}")
    ap.add_argument("--crossover-rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-ray", action="store_true",
                    help="Enable ray parallel NOMAD calls. :contentReference[oaicite:7]{index=7}")

    # scalarization / regularization
    ap.add_argument("--w-mc", type=float, default=1.0)
    ap.add_argument("--w-ipc", type=float, default=1.0)
    ap.add_argument("--w-kr", type=float, default=1.0)
    ap.add_argument("--w-gr", type=float, default=1.0)
    ap.add_argument("--norm", choices=["ratio", "diff"], default="ratio",
                    help="ratio: candidate/baseline; diff: candidate-baseline")
    ap.add_argument("--l2-penalty", type=float, default=0.0,
                    help="Penalty on mean squared synapse change over nz edges.")
    ap.add_argument("--delta-scale", type=float, default=None,
                    help="Scale for x -> additive delta. Default = std of nz weights.")
    ap.add_argument("--clip-abs", type=float, default=None,
                    help="Optional abs clip on candidate synapse weights.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load connectome
    ce_W_bio, ce_ei, _  = load_connectome(args.ce_adj, args.ce_ei)
    ce_W_bio = ce_W_bio.astype(np.float32)
    N = ce_W_bio.shape[0]

    # nz mask excluding diagonal
    A_ce = (np.abs(ce_W_bio) > 0)
    np.fill_diagonal(A_ce, False)
    nz_flat_idx = np.flatnonzero(A_ce.ravel())
    w0 = ce_W_bio.ravel()[nz_flat_idx].copy()

    delta_scale = args.delta_scale
    if delta_scale is None:
        delta_scale = float(np.std(w0) + 1e-12)

    col_params = make_col_params(args.rho_targets, args.leaks, args.input_scales)

    # baseline scores
    baseline = run_scores_for_matrix(
        WS_K=0,
        W_bio_mat=ce_W_bio,
        ce_ei=ce_ei,
        col_params=col_params,
        device=device,
        seed_base=args.seed + 123,
    )
 
    baseline_means = {k: float(np.nanmean(v)) for k, v in baseline.items()}

    # log baseline
    save_csv(
        str(out_dir / "baseline.csv"),
        ["rho_target", "leak", "input_scale", "MC", "IPC", "KR", "GR"],
        [
            (r, l, u, float(baseline["MC"][i]), float(baseline["IPC"][i]),
             float(baseline["KR"][i]), float(baseline["GR"][i]))
            for i, (r, l, u) in enumerate(col_params)
        ],
    )

    # map x -> W_candidate
    def vec_to_W(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        Wcand = ce_W_bio.copy().ravel()
        new_vals = w0 + x * delta_scale
        if args.clip_abs is not None:
            new_vals = np.clip(new_vals, -args.clip_abs, args.clip_abs)
        Wcand[nz_flat_idx] = new_vals
        Wcand = Wcand.reshape((N, N)).astype(np.float32)
        np.fill_diagonal(Wcand, 0.0)
        return Wcand

    weights = dict(MC=args.w_mc, IPC=args.w_ipc, KR=args.w_kr, GR=args.w_gr)

    # objective for EANOMAD (maximize)
    def objective_fn(x: np.ndarray) -> float:
        Wcand = vec_to_W(x)
        res = run_scores_for_matrix(
            WS_K=0,
            W_bio_mat=Wcand,
            ce_ei=ce_ei,
            col_params=col_params,
            device=device,
            seed_base=args.seed + 9999,
        )
        cand_means = {k: float(np.nanmean(v)) for k, v in res.items()}

        # fail fast on NaNs
        if not all(np.isfinite(cand_means[k]) for k in cand_means):
            return -1e9

        fit = 0.0
        for k, w in weights.items():
            b = baseline_means[k]
            c = cand_means[k]
            if args.norm == "ratio":
                fit += w * (c / (b + 1e-12))
            else:
                fit += w * (c - b)

        if args.l2_penalty > 0.0:
            diff = Wcand[A_ce] - ce_W_bio[A_ce]
            fit -= args.l2_penalty * float(np.mean(diff * diff))

        return float(fit)

    dim = len(w0)

    opt = EANOMAD(
        args.mode,
        population_size=args.population_size,
        dimension=dim,
        objective_fn=objective_fn,
        subset_size=args.subset_size,
        bounds=args.bounds,
        max_bb_eval=args.max_bb_eval,
        n_mutate_coords=args.n_mutate_coords,
        crossover_rate=args.crossover_rate,
        init_vec=np.zeros(dim, dtype=np.float32),  # start at biological weights
        use_ray=args.use_ray,
        seed=args.seed,
    )

    best_x, best_fit = opt.run(generations=args.generations)
    print(opt.fitness_history)

    Wbest = vec_to_W(best_x)

    # evaluate best fully and save
    best_scores = run_scores_for_matrix(
        WS_K=0,
        W_bio_mat=Wbest,
        ce_ei=ce_ei,
        col_params=col_params,
        device=device,
        seed_base=args.seed + 4242,
    )

    np.save(out_dir / "best_Wbio.npy", Wbest)
    np.save(out_dir / "best_x.npy", np.asarray(best_x, dtype=np.float32))

    save_csv(
        str(out_dir / "best_scores.csv"),
        ["rho_target", "leak", "input_scale", "MC", "IPC", "KR", "GR"],
        [
            (r, l, u,
             float(best_scores["MC"][i]), float(best_scores["IPC"][i]),
             float(best_scores["KR"][i]), float(best_scores["GR"][i]))
            for i, (r, l, u) in enumerate(col_params)
        ],
    )

    with open(out_dir / "best_summary.txt", "w") as f:
        f.write(f"best_fit\t{best_fit}\n")
        for k in ("MC", "IPC", "KR", "GR"):
            f.write(f"{k}_baseline_mean\t{baseline_means[k]}\n")
            f.write(f"{k}_best_mean\t{float(np.nanmean(best_scores[k]))}\n")

if __name__ == "__main__":
    main()
