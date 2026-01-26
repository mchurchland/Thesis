#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invariance sweeps for C. elegans reservoirs.
   - CSV: <out_dir>/invariance_variants.csv  (with 'src' column)
   - CSV: <out_dir>/bio_vs_shuffle_invariance.csv  (with 'src' column)
     Variants:
         * real                 (C. elegans adjacency, bio weights)
         * shuffle_weights      (C. elegans weight multiset shuffled across existing nonzeros)
         * cel_randN            (CE adjacency, Gaussian weights)
         * er_randN             (Std ESN, directed ER, Gaussian weights; nnz matched to CE)
         * ws_p01_randN         (WS p=0.1, Gaussian weights; nnz matched to CE)
         * conn_shuf            (CE weight multiset on degree-shuffled connections), repeated n_conn_shuf times
         * local_sign           (CE adjacency; preserve sign pattern, replace magnitudes with N(0,1))
"""

import os
import argparse
import itertools

import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe for headless cluster

import torch

from reservoir_variants import VARIANT_REGISTRY, VariantContext, run_variant, save_rows

# ---- repo helpers (reuse your utils/stats) ----
from util.util import load_connectome

# =================== Defaults (match your diagnostics) ===================

WS_K = 40  # signature compatibility with util.build_reservoir

SWEEP_SR   = [0.6, 0.8, 0.95, 1.05]
SWEEP_LEAK = [0.6, 0.8, 1.0]
SWEEP_U    = [0.1, 0.5, 1.0, 1.5]


# =================== Core helpers ===================


def _build_col_params(
    sr_grid: list[float],
    leak_grid: list[float],
    u_grid: list[float],
) -> list[tuple[float, float, float]]:
    # Cartesian product of (spectral radius target, leak, input scale)
    return [(sr, leak, u) for sr, leak, u in itertools.product(sr_grid, leak_grid, u_grid)]
    ##essentially just do all combinations of the three lists


def _split_indices(n_total: int, split: int, rank: int) -> list[int]:
    """
    Return the indices this rank should handle (array-job friendly).

    Supports both:
      - 0-based ranks in [0, split-1]
      - 1-based ranks in [1, split]  (e.g. SLURM_ARRAY_TASK_ID directly)
    """
    if n_total == 0:
        return []

    if split <= 1:
        # single job handles entire grid
        return list(range(n_total))

    # Normalize rank
    if not (0 <= rank < split):
        # try to interpret as 1-based
        if 1 <= rank <= split:
            rank = rank - 1
        else:
            raise ValueError(
                f"--rank must be in [0, {split-1}] or [1, {split}] for --split={split}, got {rank}"
            )

    base = n_total // split
    rem = n_total % split

    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


def _pick_device(cuda_index: int | None) -> torch.device:
    if cuda_index is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda_index >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_ctx(
    job_key: str,
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    device: torch.device,
    seed: int,
    sid: int,
    er_p: float,
    ws_p: float,
    src_tag: str,
) -> VariantContext:
    if job_key not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant key: {job_key}")
    return VariantContext(
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        ws_k=WS_K,
        col_params=col_params,
        device=device,
        seed=seed,
        sid=sid,
        er_p=er_p,
        ws_p=ws_p,
        src_tag=src_tag,
    )


def _run_and_save(job_key: str, ctx: VariantContext, out_dir: str, csv_name: str, append: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    rows = run_variant(VARIANT_REGISTRY[job_key], ctx)
    save_rows(out_csv, rows, append=append)


def parse_args():
    p = argparse.ArgumentParser(
        description="Invariance sweeps for C. elegans reservoirs (cluster-friendly)."
    )
    # What to run
    p.add_argument(
        "--job",
        choices=sorted(VARIANT_REGISTRY.keys()),
        required=True,
        help="Select a single variant per invocation; use array jobs to sweep sids etc.",
    )
    p.add_argument("--out-dir", required=True, help="Output directory for CSVs.")

    # Connectome paths
    p.add_argument(
        "--ce-adj",
        default=None,
        help="Path to C. elegans adjacency matrix (n x n).",
    )
    p.add_argument(
        "--ce-ei",
        default=None,
        help="Path to C. elegans E/I matrix (n x n).",
    )

    # IO / provenance
    p.add_argument("--csv-name", default=None, help="Optional CSV file name override.")
    p.add_argument("--src-tag", default="chunk_0", help="Provenance tag stored in 'src'.")

    # RNG / run ids
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument(
        "--sid",
        type=int,
        default=1,
        help="Shuffle/run id for variants that need it.",
    )
    p.add_argument(
        "--n-shuffles",
        type=int,
        default=1,
        help="Repeat count for shuffle-style jobs.",
    )

    # Graph model params
    p.add_argument(
        "--er-p",
        type=float,
        default=0.1,
        help="ER edge probability for er_randN.",
    )
    p.add_argument(
        "--ws-p",
        type=float,
        default=0.1,
        help="WS rewiring probability for ws_p01_randN (accepted for compatibility).",
    )


    # Array-job partitioning of the parameter grid
    p.add_argument(
        "--split",
        type=int,
        default=1,
        help="Total partitions of the param grid (e.g. number of SLURM array tasks).",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=0,
        help="This process's partition index (0-based or 1-based; see _split_indices).",
    )

    # Repeat partitioning (run multiple seeds per job)
    p.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Total repeats across all tasks (each repeat uses a new seed).",
    )
    p.add_argument(
        "--repeat-split",
        type=int,
        default=1,
        help="Total partitions of repeats (e.g. number of SLURM array tasks).",
    )
    p.add_argument(
        "--repeat-rank",
        type=int,
        default=0,
        help="This task's repeat partition index (0-based or 1-based).",
    )
    p.add_argument(
        "--repeat-seed-stride",
        type=int,
        default=100000,
        help="Seed increment per repeat.",
    )

    # Device
    p.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="CUDA device index; omit for auto.",
    )


    return p.parse_args()


# ------------------------------ main -----------------------------------------

def main():
    args = parse_args()

    # Build parameter grid and optionally slice for array jobs
    sr_grid   = SWEEP_SR
    leak_grid = SWEEP_LEAK
    u_grid    = SWEEP_U
    col_params_full = _build_col_params(sr_grid, leak_grid, u_grid)

    # Partition the grid so each array task does a subset
    idxs = _split_indices(len(col_params_full), args.split, args.rank)
    col_params = [col_params_full[i] for i in idxs]

    device = _pick_device(args.cuda)

    # Load connectome (most helpers require CE bio matrix and optional E/I)
    # Priority:
    #   1) explicit --ce-adj + --ce-ei
    #   2) --ce-path
    #   3) load_connectome() default
    if args.ce_adj is not None and args.ce_ei is not None:
            ce_W_bio, ce_ei, _  = load_connectome(args.ce_adj, args.ce_ei)
            print(ce_W_bio,ce_ei)
    else:
        raise ValueError(
            "You must pass both --ce-adj and --ce-ei, or --ce-path to load_connectome()."
        )
        

    # Decide CSV name default per job if not overridden
    if args.csv_name is not None:
        csv_name = args.csv_name
    else:
        if args.job in ("real", "shuffle_weights"):
            csv_name = "bio_vs_shuffle_invariance.csv"
        elif args.job in ("cel_randN", "er_randN", "ws_p01_randN"):
            csv_name = "cel_variants.csv"
        else:
            csv_name = "invariance_variants.csv"

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Partition repeats across tasks
    repeat_ids = _split_indices(args.n_repeats, args.repeat_split, args.repeat_rank)
    if not repeat_ids:
        print(
            f"[INFO] no repeats assigned to rank={args.repeat_rank} "
            f"(repeat-split={args.repeat_split}, n-repeats={args.n_repeats})"
        )
        return

    # ---------------- job dispatch ----------------
    job_key = args.job

    def _sid_base(rep_idx: int) -> int:
        if job_key in ("shuffle_weights", "conn_shuf", "local_sign"):
            return args.sid + rep_idx * max(1, args.n_shuffles)
        if job_key == "real":
            return args.sid + rep_idx
        return -1

    for rep_pos, rep_idx in enumerate(repeat_ids):
        append_base = rep_pos > 0
        seed_base = args.seed + rep_idx * args.repeat_seed_stride
        set_seed(seed_base)
        sid_base = _sid_base(rep_idx)

        if job_key == "shuffle_weights":
            for j in range(args.n_shuffles):
                sid = sid_base if args.n_shuffles == 1 else (sid_base + j)
                ctx = _build_ctx(
                    job_key,
                    WS_K,
                    ce_W_bio,
                    ce_ei,
                    col_params,
                    device,
                    seed=seed_base + 7_000 * j,
                    sid=sid,
                    er_p=args.er_p,
                    ws_p=args.ws_p,
                    src_tag=args.src_tag,
                )
                _run_and_save(job_key, ctx, out_dir, csv_name, append=(append_base or j > 0))
            continue

        if job_key == "conn_shuf":
            for j in range(args.n_shuffles):
                sid = sid_base if args.n_shuffles == 1 else (sid_base + j)
                ctx = _build_ctx(
                    job_key,
                    WS_K,
                    ce_W_bio,
                    ce_ei,
                    col_params,
                    device,
                    seed=seed_base + 9_000 * j,
                    sid=sid,
                    er_p=args.er_p,
                    ws_p=args.ws_p,
                    src_tag=args.src_tag,
                )
                _run_and_save(job_key, ctx, out_dir, csv_name, append=(append_base or j > 0))
            continue

        if job_key == "local_sign":
            for j in range(args.n_shuffles):
                sid = sid_base if args.n_shuffles == 1 else (sid_base + j)
                ctx = _build_ctx(
                    job_key=job_key,
                    WS_K=WS_K,
                    ce_W_bio=ce_W_bio,
                    ce_ei=None,
                    col_params=col_params,
                    device=device,
                    seed=seed_base + 9_000 * j,
                    sid=sid,
                    er_p=args.er_p,
                    ws_p=args.ws_p,
                    src_tag=args.src_tag,
                )
                _run_and_save(job_key, ctx, out_dir, csv_name, append=(append_base or j > 0))
            continue

        ctx = _build_ctx(
            job_key,
            WS_K,
            ce_W_bio,
            ce_ei,
            col_params,
            device,
            seed=seed_base,
            sid=sid_base,
            er_p=args.er_p,
            ws_p=args.ws_p,
            src_tag=args.src_tag,
        )
        _run_and_save(job_key, ctx, out_dir, csv_name, append=append_base)


# ------------------------------ CLI entry ------------------------------------

if __name__ == "__main__":
    main()
