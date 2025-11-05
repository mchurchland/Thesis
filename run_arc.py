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

# ---- repo helpers (reuse your utils/stats) ----
from util.util import load_connectome, build_reservoir
from network_stats.stats import compute_IPC, compute_KR, compute_GR, compute_MC
from util.util import degree_matched_shuffle_directed
from network_stats.run_one import run_one
def save(out_csv: str, rows: list[tuple[str,int,float,float,float,float,float,float,float,str]]):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode","shuffle_id","rho_target","leak","input_scale","MC","IPC","KR","GR","src"])
        w.writerows(rows)
##these can be combined into a single function, but I left them separate for now
def _run_variant_row(   WS_K: int,
                        feature_conn: str,
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
def _run_row_for_matrix(WS_K: int,
                        W_bio_mat: np.ndarray,
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
    WS_K: int,
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
    res = _run_row_for_matrix(WS_K,Wsh, col_params, ce_ei, device, seed_base=seed+9999+sid)

    for (rho_t, leak, u), mc, ipc, kr, gr in zip(col_params, res["MC"], res["IPC"], res["KR"], res["GR"]):
        rows.append(("shuffle", sid, rho_t, leak, u, float(mc), float(ipc), float(kr), float(gr), src_tag))
    save(out_csv, rows)
def run_one_real(
    WS_K: int,
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
    real = _run_row_for_matrix(WS_K,ce_W_bio, col_params, ce_ei, device, seed_base=seed+123)

    # Append real CE
    for (rho_t, leak, u), mc, ipc, kr, gr in zip(col_params, real["MC"], real["IPC"], real["KR"], real["GR"]):
        rows.append(("real", nid, rho_t, leak, u, float(mc), float(ipc), float(kr), float(gr), src_tag))
    save(out_csv, rows)
def run_one_cel_randN(
    WS_K: int,
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
    rows += _run_variant_row(WS_K,"cel", "rand_gauss",col_params,device, ce_W_bio,\
                                  "cel+randN", -1, ce_W_bio,ce_ei=ce_ei, src_tag=src_tag,seed_base=seed + 10_000)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    save(out_csv, rows)
def run_one_esn_er_randN(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    er_p: float = 0.1,
    seed: int = 0,
    csv_name: str = "cel_variants.csv",
    src_tag: str = "chunk_0"):
    rows = []
    rows += _run_variant_row(WS_K,f"er_p={er_p}", "rand_gauss",col_params,device, None,\
                                  "er+randN", -1, ce_W_bio,ce_ei=ce_ei, src_tag=src_tag,seed_base=seed + 20_000)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    save(out_csv, rows)
def run_one_ws_p0_1_randN(
    WS_K: int,
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
    rows += _run_variant_row(WS_K,"ws_p=0.1", "rand_gauss",col_params,device, None,\
                                  "ws_p0.1+randN", -1, ce_W_bio,ce_ei=ce_ei, src_tag=src_tag,seed_base=seed + 20_000)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)
    save(out_csv, rows)
def run_one_celW_connShuf(
    WS_K: int,
    ce_W_bio: np.ndarray,
    ce_ei: np.ndarray | None,
    col_params: list[tuple[float, float, float]],
    out_dir: str,
    device: torch.device,
    *,
    sid: int = 1,
    seed: int = 0,
    csv_name: str = "invariance_variants.csv",
    src_tag: str = "chunk_0",
):
    """
    Degree-matched shuffle of CE adjacency; reassign the CE weight multiset to the
    shuffled edges (no self-loops), then measure invariance across col_params.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, csv_name)

    rng = np.random.default_rng(seed + 40_000 + sid)
    # Build CE adjacency (no diagonal)
    A_ce = (np.abs(ce_W_bio) > 0)
    np.fill_diagonal(A_ce, False)
    nnz_target = int(A_ce.sum())
    ce_weights_all = ce_W_bio[A_ce].astype(np.float32)

    # Degree-matched directed shuffle
    As = degree_matched_shuffle_directed(A_ce.astype(np.float32), tries=20_000, rng=rng).astype(bool)
    # Assign permuted weights to shuffled edges; ensure same nnz
    if int(As.sum()) != nnz_target:
        # Defensive: fall back to masking equal count
        # Keep only the first nnz_target edges in As' flattened order
        # by zeroing extras (rare, but protects correctness).
        flat_idx = np.flatnonzero(As.ravel())
        if len(flat_idx) > nnz_target:
            As = As.ravel()
            As[flat_idx[nnz_target:]] = False
            As = As.reshape(A_ce.shape)

    Wsh = np.zeros_like(ce_W_bio, dtype=np.float32)
    perm = rng.permutation(len(ce_weights_all))
    Wsh[As] = ce_weights_all[perm][:nnz_target]
    np.fill_diagonal(Wsh, 0.0)

    rows = _run_variant_row(WS_K,
        feature_conn="cel",
        feature_weights="bio",
        col_params=col_params,
        device=device,
        custom_ce=Wsh,
        mode_label="celW+connShuf",
        shuffle_id=sid,
        ce_W_bio=ce_W_bio,
        ce_ei=ce_ei,
        src_tag=src_tag,
        seed_base=seed + 50_000 + sid * 911,
    )

    save(out_csv, rows)



def _shuffle_ce_weights(Wbio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = Wbio.copy().astype(np.float32)
    nz = np.nonzero(W)
    vals = W[nz].copy()
    rng.shuffle(vals)
    W[nz] = vals
    return W

