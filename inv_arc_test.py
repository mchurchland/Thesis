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
         * sign_test            (CE adjacency; flip the sign on fraction(s) via --sign-flip-frac, 
         * all                  (run every variant above; append to avoid clobbering)
"""

import os
import argparse
import itertools

import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe for headless cluster

import torch

from reservoir_variants import VARIANT_KEYS, VariantContext, run_variant, save_rows

# ---- repo helpers (reuse your utils/stats) ----
from util.util import load_connectome

# =================== Defaults (match your diagnostics) ===================

WS_K = 10  # signature compatibility with util.build_reservoir

ALL_JOB_KEYS = ( ## I need to change the names anyways lets do this inclass local_sign+binary should be local
    "real",            #x
    #"shuffle_weights",
    "cel_randN", #x
    "er_randN", #x
    "ws_p01_randN", #x
    #"conn_shuf", 
    "local_sign", #x
    #"conn_shuf_only",
    #"cel_sample", 
    "local_sign+flat", #x 
    #"local_sign+sample", 
    "local_sign+binary", #x
    "global_sign_pres", #x
    "binary_base", #x
    #"binary_base_topology_shuffle", #x
    #"binary+shuffle", #x
    #"binary+conshuffle+wshuffle", #x
    #"sign_test_og_cel",
)

#ALL_JOB_KEYS = (
#    "real",
#    "er_randN",
#    "local_sign",
#)

TOPOLOGY_SHUFFLE_JOB_KEYS = (
    "real",
    "shuffle_weights",
    "conn_shuf_only",
    "conn_shuf",
    "local_sign+binary", ##this is localsign+signed binary weights
    "binary_base", ## unsigned binary
    "binary_base_topology_shuffle", ##unsigned binary w shuffle
    "binary+shuffle",
    "binary+conshuffle+wshuffle",
)

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
    per_neg: float | None = None,
    alpha: float | None = None,
) -> VariantContext:
    if job_key not in VARIANT_KEYS:
        if (not job_key.startswith("sign_test")) and (not job_key.startswith("weight_test")):
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
        per_neg=per_neg,
        alpha=alpha
    )


def _run_and_save(job_key: str, ctx: VariantContext, out_dir: str, csv_name: str, append: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    rows = run_variant(job_key, ctx)
    save_rows(out_csv, rows, append=append)


def parse_args():
    p = argparse.ArgumentParser(
        description="Invariance sweeps for C. elegans reservoirs (cluster-friendly)."
    )
    # What to run
    p.add_argument(
        "--job",
        choices=sorted(VARIANT_KEYS) + ["all", "all_topology_shuffle"],
        required=True,
        help="Select a single variant, 'all', or 'all_topology_shuffle' for the topology-shuffle section suite.",
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
    "--rho-test",
    action="store_true",
    help="Run the extended rho sweep.",
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
    p.add_argument(
        "--sign-flip-frac",
        type=float,
        nargs="+",
        default=0.0,
        help="Space-separated fraction(s) of CE edges whose sign is flipped for sign_test (0 <= frac <= 1).",
    )
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=0.0,
        help="alphas for the weight test",
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
    sign_flip_fracs = list(args.sign_flip_frac if isinstance(args.sign_flip_frac, (list, tuple)) else [args.sign_flip_frac])
    alphas = list(args.alphas if isinstance(args.alphas, (list, tuple)) else [args.alphas])
    for frac in sign_flip_fracs:
        if not (0.0 <= frac <= 1.0):
            raise ValueError("--sign-flip-frac values must be between 0 and 1 inclusive for sign_test.")
    # Build parameter grid and optionally slice for array jobs
    sr_grid   = SWEEP_SR if not args.rho_test else [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0, 4.0, 10.0]
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
    else:
        raise ValueError(
            "You must pass both --ce-adj and --ce-ei, or --ce-path to load_connectome()."
        )
        

    # Decide CSV name default per job if not overridden
    if args.csv_name is not None:
        csv_name = args.csv_name
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
    def _run_job(job_key: str, *, append_start: bool):
        def _sid_base(rep_idx: int) -> int:
            return args.sid + rep_idx


        for rep_pos, rep_idx in enumerate(repeat_ids):
            append_base = append_start or rep_pos > 0
            seed_base = args.seed + rep_idx * args.repeat_seed_stride
            set_seed(seed_base)
            sid_base = _sid_base(rep_idx)

            if job_key == "shuffle_weights":
                    ctx = _build_ctx(
                        job_key,
                        WS_K,
                        ce_W_bio,
                        None,
                        col_params,
                        device,
                        seed=seed_base,
                        sid=sid_base,
                        er_p=args.er_p,
                        ws_p=args.ws_p,
                        src_tag=args.src_tag,
                    )
                    _run_and_save(job_key, ctx, out_dir, csv_name, append=append_base)
                    continue

            if job_key == "conn_shuf":
                    ctx = _build_ctx(
                        job_key,
                        WS_K,
                        ce_W_bio,
                        None,
                        col_params,
                        device,
                        seed=seed_base,
                        sid=sid_base,
                        er_p=args.er_p,
                        ws_p=args.ws_p,
                        src_tag=args.src_tag,
                    )
                    _run_and_save(job_key, ctx, out_dir, csv_name, append=append_base)
                    continue

            if job_key == "local_sign":
                    ctx = _build_ctx(
                        job_key=job_key,
                        WS_K=WS_K,
                        ce_W_bio=ce_W_bio,
                        ce_ei=None,
                        col_params=col_params,
                        device=device,
                        seed=seed_base,
                        sid=sid_base,
                        er_p=args.er_p,
                        ws_p=args.ws_p,
                        src_tag=args.src_tag,
                        per_neg=None,
                    )
                    _run_and_save(job_key, ctx, out_dir, csv_name, append=append_base)
                    continue
            if job_key == "weight_test":
                for frac_idx, alpha in enumerate(alphas):
                    ctx = _build_ctx(
                        job_key + str(alpha),
                        WS_K,
                        ce_W_bio,
                        None,
                        col_params,
                        device,
                        seed=seed_base,
                        sid=sid_base,
                        er_p=args.er_p,
                        ws_p=args.ws_p,
                        src_tag=args.src_tag,
                        alpha=alpha
                    )
                    _run_and_save(
                        job_key + str(alpha),
                        ctx,
                        out_dir,
                        csv_name,
                        append=(append_base or frac_idx > 0),
                    )
                continue
            if job_key == "sign_test_cel":
                for frac_idx, frac in enumerate(sign_flip_fracs):
                    ctx = _build_ctx(
                        job_key + str(frac),
                        WS_K,
                        ce_W_bio,
                        None,
                        col_params,
                        device,
                        seed=seed_base,
                        sid=sid_base,
                        er_p=args.er_p,
                        ws_p=args.ws_p,
                        src_tag=args.src_tag,
                        per_neg=frac
                    )
                    _run_and_save(
                        job_key + str(frac),
                        ctx,
                        out_dir,
                        csv_name,
                        append=(append_base or frac_idx > 0),
                    )
                continue
            if job_key == "sign_test_og_cel":
                frac = (len(np.where(ce_W_bio<0)[0])/ (len(np.where(ce_W_bio> 0)[0]) + len(np.where(ce_W_bio<0)[0])) ) 
                # Build a local list for this run; do not mutate sign_flip_fracs in-place.
                fracs_local = [frac]
                for base_frac in sign_flip_fracs:
                    if not np.isclose(base_frac, frac):
                        fracs_local.append(base_frac)
                for frac_idx, frac in enumerate(fracs_local):
                    ctx = _build_ctx(
                        job_key + str(frac),
                        WS_K,
                        ce_W_bio,
                        None,
                        col_params,
                        device,
                        seed=seed_base,
                        sid=sid_base,
                        er_p=args.er_p,
                        ws_p=args.ws_p,
                        src_tag=args.src_tag,
                        per_neg=frac
                    )
                    _run_and_save(
                        job_key + str(frac),
                        ctx,
                        out_dir,
                        csv_name,
                        append=(append_base or frac_idx > 0),
                    )
                continue
            if job_key == "sign_test_er":
                for frac_idx, frac in enumerate(sign_flip_fracs):
                    ctx = _build_ctx(
                        job_key + str(frac),
                        WS_K,
                        ce_W_bio,
                        None,
                        col_params,
                        device,
                        seed=seed_base,
                        sid=sid_base,
                        er_p=args.er_p,
                        ws_p=args.ws_p,
                        src_tag=args.src_tag,
                        per_neg=frac
                    )
                    _run_and_save(
                        job_key + str(frac),
                        ctx,
                        out_dir,
                        csv_name,
                        append=(append_base or frac_idx > 0),
                    )
                continue
            ctx = _build_ctx(
                job_key,
                WS_K,
                ce_W_bio,
                None,
                col_params,
                device,
                seed=seed_base,
                sid=sid_base,
                er_p=args.er_p,
                ws_p=args.ws_p,
                src_tag=args.src_tag,
                per_neg=None
            )
            _run_and_save(job_key, ctx, out_dir, csv_name, append=append_base)

    if args.job == "all":
        out_csv = os.path.join(out_dir, csv_name)
        append_across = os.path.exists(out_csv)
        for idx, job_key in enumerate(ALL_JOB_KEYS):
            _run_job(job_key, append_start=(append_across or idx > 0))
            append_across = True
    elif args.job == "all_topology_shuffle":
        out_csv = os.path.join(out_dir, csv_name)
        append_across = os.path.exists(out_csv)
        for idx, job_key in enumerate(TOPOLOGY_SHUFFLE_JOB_KEYS):
            _run_job(job_key, append_start=(append_across or idx > 0))
            append_across = True
    else:
        _run_job(args.job, append_start=False)


# ------------------------------ CLI entry ------------------------------------

if __name__ == "__main__":
    main()
