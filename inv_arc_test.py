#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invariance sweeps for C. elegans reservoirs.
   - CSV: <out_dir>/invariance_variants.csv  (with 'src' column)
   - CSV: <out_dir>/bio_vs_shuffle_invariance.csv  (with 'src' column)
     Variants:
         * real                 (C. elegans adjacency, bio weights)
            * celW+wshuf      (C. elegans weight multiset on degree-shuffled connections)
       * cel+randN            (CE adjacency, Gaussian weights)
       * er+randN             (Std ESN, directed ER, Gaussian weights; nnz matched to CE)
       * ws_p0.1+randN        (WS p=0.1, Gaussian weights; nnz matched to CE)
       * celW+connShuf        (CE weight multiset on degree-shuffled connections), repeated n_conn_shuf times
"""

import os
import csv
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import itertools
from pathlib import Path
from typing import Iterable
import torch
from torch import Tensor
from run_arc import run_one_real, run_one_shuf_weights, run_one_cel_randN,\
             run_one_esn_er_randN, run_one_ws_p0_1_randN, run_one_celW_connShuf

# ---- repo helpers (reuse your utils/stats) ----
from util.util import load_connectome, build_reservoir
from network_stats.stats import compute_IPC, compute_KR, compute_GR, compute_MC
from util.util import degree_matched_shuffle_directed
from network_stats.run_one import run_one
# =================== Defaults (match your diagnostics) ===================

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
WS_K           = 40  # signature compatibility with util.build_reservoir

SWEEP_SR       = [0.6, 0.8, 0.95, 1.05,1.5, 2.0] ## 1.05,2.0 added
SWEEP_LEAK     = [0.1,0.2,0.6, 0.8, 1.0] ## 0.1,0.2 added
SWEEP_U        = [0.1, 0.5, 1.0, 1.5, 3.0, 5.0]  # input scale 3.0,5.0 added

# =================== Core helpers ===================
def _as_float_list(s: str | None, default: Iterable[float]) -> list[float]:
    if s is None or s.strip() == "":
        return list(default)
    return [float(x) for x in s.replace(" ", "").split(",")]

def _build_col_params(sr_grid: list[float], leak_grid: list[float], u_grid: list[float]) -> list[tuple[float, float, float]]:
    # Cartesian product of (spectral radius target, leak, input scale)
    return [(sr, leak, u) for sr, leak, u in itertools.product(sr_grid, leak_grid, u_grid)]

def _split_indices(n_total: int, split: int, rank: int) -> list[int]:
    # Return the indices this rank should handle (array-job friendly).
    # Simple contiguous chunking.
    if split <= 1:
        return list(range(n_total))
    if not (0 <= rank < split):
        raise ValueError(f"--rank must be in [0, {split-1}] for --split={split}")
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







def parse_args():
    p = argparse.ArgumentParser(
        description="Invariance sweeps for C. elegans reservoirs (cluster-friendly)."
    )
    # What to run
    p.add_argument(
        "--job",
        choices=[
            "real",              # CE bio weights on CE adjacency
            "shuffle_weights",   # CE adjacency, CE weights shuffled across nonzeros
            "cel_randN",         # CE adjacency, Gaussian weights
            "er_randN",          # ER directed, Gaussian weights (nnz matched via util)
            "ws_p01_randN",      # WS p=0.1, Gaussian weights (nnz matched via util)
            "conn_shuf",         # CE weights on degree-matched shuffled adjacency
        ],
        required=True,
        help="Select a single variant per invocation; use array jobs to sweep sids etc.",
    )
    p.add_argument("--out-dir", required=True, help="Output directory for CSVs.")
    p.add_argument("--csv-name", default=None, help="Optional CSV file name override.")
    p.add_argument("--src-tag", default="chunk_0", help="Provenance tag stored in 'src'.")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument("--sid", type=int, default=1, help="Shuffle/run id for variants that need it.")
    p.add_argument("--n-shuffles", type=int, default=1, help="Repeat count for shuffle-style jobs.")
    p.add_argument("--er-p", type=float, default=0.1, help="ER edge probability for er_randN.")
    # Grids
    p.add_argument("--sr-grid", default=None, help=f"Comma list (default: {SWEEP_SR}).")
    p.add_argument("--leak-grid", default=None, help=f"Comma list (default: {SWEEP_LEAK}).")
    p.add_argument("--u-grid", default=None, help=f"Comma list (default: {SWEEP_U}).")
    # Array-job partitioning of the parameter grid
    p.add_argument("--split", type=int, default=1, help="Total partitions of the param grid.")
    p.add_argument("--rank", type=int, default=0, help="This process's partition index [0..split-1].")
    # Device
    p.add_argument("--cuda", type=int, default=None, help="CUDA device index; omit for auto.")
    # Connectome loading
    p.add_argument(
        "--ce-path",
        default=None,
        help="Optional path passed to util.load_connectome(). If omitted, that helper's default is used.",
    )
    return p.parse_args()

# ------------------------------ main -----------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Build parameter grid and optionally slice for array jobs
    sr_grid   = _as_float_list(args.sr_grid, SWEEP_SR)
    leak_grid = _as_float_list(args.leak_grid, SWEEP_LEAK)
    u_grid    = _as_float_list(args.u_grid, SWEEP_U)
    col_params_full = _build_col_params(sr_grid, leak_grid, u_grid)

    # Partition the grid so each array task does a subset
    idxs = _split_indices(len(col_params_full), args.split, args.rank)
    col_params = [col_params_full[i] for i in idxs]

    device = _pick_device(args.cuda)

    # Load connectome (most helpers require CE bio matrix and optional E/I)
    # Support both signatures: load_connectome() and load_connectome(pathlike)
    try:
        if args.ce_path is None:
            ce_W_bio, ce_ei = load_connectome()
        else:
            ce_W_bio, ce_ei = load_connectome(args.ce_path)
    except TypeError:
        # Fallback: some repos return only W
        if args.ce_path is None:
            ce_W_bio = load_connectome()
        else:
            ce_W_bio = load_connectome(args.ce_path)
        ce_ei = None

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
    os.makedirs(out_dir, exist_ok=True)(out_dir)

    if args.job == "real":
        # One pass over CE bio weights
        run_one_real(
            WS_K=WS_K,
            ce_W_bio=ce_W_bio,
            ce_ei=ce_ei,
            col_params=col_params,
            out_dir=out_dir,
            device=device,
            seed=args.seed,
            nid=args.sid,
            csv_name=csv_name,
            src_tag=args.src_tag,
        )
        return

    if args.job == "shuffle_weights":
        # Possibly repeat multiple independent shuffles for the same param subset
        for j in range(args.n_shuffles):
            sid = args.sid if args.n_shuffles == 1 else (args.sid + j)
            run_one_shuf_weights(
                WS_K=WS_K,
                ce_W_bio=ce_W_bio,
                ce_ei=ce_ei,
                col_params=col_params,
                out_dir=out_dir,
                device=device,
                seed=args.seed + 7_000 * j,
                sid=sid,
                metric="MC",  # stored anyway; keep API stable
                csv_name=csv_name,
                src_tag=args.src_tag,
            )
        return

    if args.job == "cel_randN":
        run_one_cel_randN(
            WS_K=WS_K,
            ce_W_bio=ce_W_bio,
            ce_ei=ce_ei,
            col_params=col_params,
            out_dir=out_dir,
            device=device,
            seed=args.seed,
            csv_name=csv_name,
            src_tag=args.src_tag,
        )
        return

    if args.job == "er_randN":
        run_one_esn_er_randN(
            WS_K=WS_K,
            ce_W_bio=ce_W_bio,
            ce_ei=ce_ei,
            col_params=col_params,
            out_dir=out_dir,
            device=device,
            er_p=args.er_p,
            seed=args.seed,
            csv_name=csv_name,
            src_tag=args.src_tag,
        )
        return

    if args.job == "ws_p01_randN":
        run_one_ws_p0_1_randN(
            WS_K=WS_K,
            ce_W_bio=ce_W_bio,
            ce_ei=ce_ei,
            col_params=col_params,
            out_dir=out_dir,
            device=device,
            seed=args.seed,
            csv_name=csv_name,
            src_tag=args.src_tag,
        )
        return

    if args.job == "conn_shuf":
        # Degree-matched connection shuffles; optionally repeat
        for j in range(args.n_shuffles):
            sid = args.sid if args.n_shuffles == 1 else (args.sid + j)
            run_one_celW_connShuf(
                WS_K=WS_K,
                ce_W_bio=ce_W_bio,
                ce_ei=ce_ei,
                col_params=col_params,
                out_dir=out_dir,
                device=device,
                sid=sid,
                seed=args.seed + 9_000 * j,
                csv_name=csv_name,
                src_tag=args.src_tag,
            )
        return

    raise RuntimeError(f"Unhandled job: {args.job}")

# ------------------------------ CLI entry ------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail noisily on clusters (stderr) with a nonzero code.
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(2)