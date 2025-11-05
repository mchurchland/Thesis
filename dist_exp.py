#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invariance sweeps for C. elegans reservoirs.

This script produces two kinds of outputs:

1) CEL+bioW vs weight-shuffled nulls (same test you already had)
   - CSV: <out_dir>/bio_vs_shuffle_invariance.csv  (with 'src' column)
   - FIG: <out_dir>/bio_invariance_hist_<METRIC>.png

2) Additional variants (single rows or multiple rewires) saved to one CSV:
   - CSV: <out_dir>/invariance_variants.csv  (with 'src' column)
     Variants:
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

import torch
from torch import Tensor

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

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save(out_csv: str, rows: list[tuple[str,int,float,float,float,float,float,float,float,str]]):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode","shuffle_id","rho_target","leak","input_scale","MC","IPC","KR","GR","src"])
        w.writerows(rows)
def _shuffle_ce_weights(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = Wbio.copy().astype(np.float32)
    nz = np.nonzero(W)
    vals = W[nz].copy()
    rng.shuffle(vals)
    W[nz] = vals
    return W





##these can be combined into a single function, but I left them separate for now
def _run_variant_row(feature_conn: str,
                        feature_weights: str,
                        col_params: list[tuple[float, float, float]],
                        device: torch.device,
                        custom_ce: np.ndarray | None,
                        mode_label: str,
                        shuffle_id: int = -1,
                        ce_W_bio: np.ndarray | None = None,
                        ce_ei: np.ndarray | None = None,
                        src_tag: str = "chunk_0",
                        seed_base: int = 0) -> list[tuple]:
        rows_local = []
        Nloc = (custom_ce.shape[0] if (custom_ce is not None and feature_conn == "cel") else ce_W_bio.shape[0])
        for ci, (target_sr, leak, in_scale) in enumerate(col_params):
            try:
                Wt, Win, _, _, _ = build_reservoir(
                    feature_conn=feature_conn,
                    feature_weights=feature_weights,
                    feature_dale="none",
                    target_sr=target_sr,
                    N=Nloc,
                    ce_W_bio=(custom_ce if feature_conn == "cel" else ce_W_bio),
                    ce_ei=ce_ei,
                    ws_k=WS_K,
                    input_scale=in_scale,
                    seed=seed_base + ci * 101,
                    drive_idx=None,
                    # For ER/WS, keep edge-count comparable to CE when available
                    nnz_target=int((np.abs(ce_W_bio) > 0).sum() - ce_W_bio.shape[0]) if feature_conn != "cel" else None
                )
                Wt, Win = Wt.to(device), Win.to(device)
                sc = run_one(Wt, Win, leak, device)
                rows_local.append((
                    mode_label, shuffle_id, target_sr, leak, in_scale,
                    float(sc["MC"]), float(sc["IPC"]), float(sc["KR"]), float(sc["GR"]), src_tag
                ))
            except Exception:
                rows_local.append((mode_label, shuffle_id, target_sr, leak, in_scale,
                                   np.nan, np.nan, np.nan, np.nan, src_tag))
        return rows_local
def _run_row_for_matrix(W_bio_mat: np.ndarray,
                        col_params: list[tuple[float,float,float]],
                        ce_ei: np.ndarray | None,
                        device: torch.device,
                        seed_base: int = 0) -> dict[str, np.ndarray]:
    scores = {k: [] for k in ("MC","IPC","KR","GR")}
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
                seed=seed_base + ci*101,
                drive_idx=None,
                nnz_target=None
            )
            Wt  = Wt.to(device)
            Win = Win.to(device)
            sc = run_one(Wt, Win, leak, device)
        except Exception:
            sc = dict(MC=np.nan, IPC=np.nan, KR=np.nan, GR=np.nan)

        for k in scores: scores[k].append(float(sc[k]))
    return {k: np.asarray(v, dtype=np.float32) for k, v in scores.items()}

def run_one_shuf_weights(
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float,float,float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    sid: int = -1,
    metric: str = "MC",
    csv_name: str = "bio_vs_shuffle_invariance.csv",
    src_tag: str = "chunk_0",
):
    assert metric in ("MC","IPC","KR","GR")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)

    rng = np.random.default_rng(seed)

    rows = []
    Wsh = _shuffle_ce_weights(ce_W_bio, rng)
    res = _run_row_for_matrix(Wsh, col_params, ce_ei, device, seed_base=seed+9999+sid)

    for (rho_t, leak, u), mc, ipc, kr, gr in zip(col_params, res["MC"], res["IPC"], res["KR"], res["GR"]):
        rows.append(("shuffle", sid, rho_t, leak, u, float(mc), float(ipc), float(kr), float(gr), src_tag))
    save(out_csv, rows)
def run_one_real(
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float,float,float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    nid: int = 1,
    csv_name: str = "bio_vs_shuffle_invariance.csv",
    src_tag: str = "chunk_0",
):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    rows = []
    rng = np.random.default_rng(seed)

    # Real CE row
    real = _run_row_for_matrix(ce_W_bio, col_params, ce_ei, device, seed_base=seed+123)

    # Append real CE
    for (rho_t, leak, u), mc, ipc, kr, gr in zip(col_params, real["MC"], real["IPC"], real["KR"], real["GR"]):
        rows.append(("real", nid, rho_t, leak, u, float(mc), float(ipc), float(kr), float(gr), src_tag))
    save(out_csv, rows)
def run_one_cel_randN(
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0"):
    rows = []
    rows += _run_variant_row("cel", "rand_gauss",col_params,device, ce_W_bio,\
                                  "cel+randN", -1, ce_W_bio,ce_ei=ce_ei, src_tag=src_tag,seed_base=seed + 10_000)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    save(out_csv, rows)
def run_one_esn_er_randN(
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0"):
    rows = []
    rows += _run_variant_row("er_p={er_p}", "rand_gauss",col_params,device, None,\
                                  "er+randN", -1, ce_W_bio,ce_ei=ce_ei, src_tag=src_tag,seed_base=seed + 20_000)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    save(out_csv, rows)
def run_one_ws_p0_1_randN(
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0"):
    rows = []
    rows += _run_variant_row("ws_p=0.1", "rand_gauss",col_params,device, None,\
                                  "ws_p0.1+randN", -1, ce_W_bio,ce_ei=ce_ei, src_tag=src_tag,seed_base=seed + 20_000)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    save(out_csv, rows)
def run_one_shuf_conn(
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    seed: int = 0,
    csv_name: str = "invariance_variants.csv",
    sid: int =1,
    er_p: float = 0.1,
    src_tag: str = "chunk_0",
):
    """
    Write rows for:
      - 'cel+randN'              : CE adjacency, Gaussian weights
      - 'er+randN'               : directed ER, Gaussian weights (nnz matched to CE)
      - 'ws_p0.1+randN'          : WS p=0.1, Gaussian weights (nnz matched to CE)
      - 'celW+connShuf'          : CE weight multiset on degree-shuffled connections (n_conn_shuf rewires)

    Output columns:
      mode, shuffle_id, rho_target, leak, input_scale, MC, IPC, KR, GR, src
    """
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    # nnz target for consistency checks
    nnz_mask = (np.abs(ce_W_bio) > 0)
    np.fill_diagonal(nnz_mask, False)
    nnz_target = int(nnz_mask.sum())

    rng = np.random.default_rng(seed + 40_000)
    A_ce = (np.abs(ce_W_bio) > 0).astype(np.float32)
    np.fill_diagonal(A_ce, 0.0)
    ce_weights_all = ce_W_bio[np.abs(ce_W_bio) > 0].astype(np.float32)

    As = degree_matched_shuffle_directed(A_ce, tries=20_000, rng=rng).astype(bool)
    if int(As.sum()) != nnz_target:
        Wsh = np.zeros_like(ce_W_bio, dtype=np.float32)
        # permute and assign the multiset of CE weights to new positions
        perm = rng.permutation(len(ce_weights_all))
        Wsh[As] = ce_weights_all[perm][:nnz_target]
        np.fill_diagonal(Wsh, 0.0)
        all_rows += _run_variant_row("cel", "bio", col_params,device,Wsh,\
                                      "celW+connShuf", sid,ce_W_bio,ce_ei, seed_base=seed + 50_000 + sid * 911)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    save(out_csv,all_rows)


# =================== CLI ===================
def parse_args():
    ap = argparse.ArgumentParser(description="Run CE-real and random-variant invariance reps.")
    ap.add_argument("--ce-adj", default="Connectome/ce_adj.npy")
    ap.add_argument("--ce-ei",  default="Connectome/ce_ei.npy")
    ap.add_argument("--out-dir", default="experiment_full_chunks/chunk_0")
    ap.add_argument("--src-tag", default="chunk_0")

    ap.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda[:idx]'")
    ap.add_argument("--threads", type=int, default=0, help="Set OMP/MKL threads if > 0")
    ap.add_argument("--seed", type=int, default=0)

    # How many independent groups per architecture
    ap.add_argument("--n-ce-real-reps", type=int, default=1000)
    ap.add_argument("--n-rand-reps",    type=int, default=1000)

    # Random graph knobs
    ap.add_argument("--er-p",  type=float, default=0.1)
    ap.add_argument("--ws-p",  type=float, default=0.1)

    # Filenames
    ap.add_argument("--ce-real-csv", default="ce_real_reps.csv")
    ap.add_argument("--variants-csv", default="invariance_variants.csv")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

    ce_W_bio, ce_ei, *_ = load_connectome(args.ce_adj, args.ce_ei)
    if ce_W_bio is None:
        raise FileNotFoundError(f"Could not load connectome at {args.ce_adj}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Construct hyperparameter columns: 4 * 3 * 4 = 48 settings
    col_params = [(s, l, u) for s in SWEEP_SR for l in SWEEP_LEAK for u in SWEEP_U]

    # CE-real replicates
    if args.n_ce_real_reps > 0:
        save_ce_real_reps(
            ce_W_bio=ce_W_bio,
            ce_ei=ce_ei,
            col_params=col_params,
            out_dir=args.out_dir,
            device=device,
            seed=args.seed,
            csv_name=args.ce_real_csv,
            src_tag=args.src_tag,
            n_ce_real_reps=args.n_ce_real_reps,
        )

    # cel+randN, er+randN, ws_p+randN replicates
    if args.n_rand_reps > 0:
        save_rand_variants_reps(
            ce_W_bio=ce_W_bio,
            ce_ei=ce_ei,
            col_params=col_params,
            out_dir=args.out_dir,
            device=device,
            seed=args.seed,
            csv_name=args.variants_csv,
            src_tag=args.src_tag,
            er_p=args.er_p,
            ws_p=args.ws_p,
            n_rand_reps=args.n_rand_reps,
        )


if __name__ == "__main__":
    main()