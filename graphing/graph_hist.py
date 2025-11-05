#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphing.py

Read ONE merged CSV (combined.ALL.csv), compute per-group dispersion,
save it, and plot normalized histograms per architecture (same style as before).

Inputs (defaults)
  experiment_full_merged/combined.ALL.csv

Outputs (defaults)
  experiment_full_merged/dispersion_by_group.ALL.csv
  experiment_full_merged/all_arch_hist_<MC|IPC|KR|GR>.png
  experiment_full_merged/mc_vs_gr_all_arch.png
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------- CLI ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Compute dispersion from a merged CSV and plot histograms.")
    ap.add_argument("--combined",
                    default="experiment_full_merged/combined.ALL.csv",
                    help="Path to merged combined CSV.")
    ap.add_argument("--out-dir",
                    default="experiment_full_merged",
                    help="Output directory for dispersion CSV and figures.")
    ap.add_argument("--bins", type=int, default=40,
                    help="Histogram bins for dispersion plots.")
    ap.add_argument("--scatter-alpha", type=float, default=0.55,
                    help="Alpha for MC-vs-GR scatter.")
    return ap.parse_args()


# ------------------------- utilities -------------------------

def _safe_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{root}.v{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["mode","shuffle_id","rho_target","leak","input_scale","MC","IPC","KR","GR","src"]
    for c in needed:
        if c not in df.columns:
            if c in ("MC","IPC","KR","GR"):
                df[c] = np.nan
            elif c == "shuffle_id":
                df[c] = -1
            elif c == "mode":
                df[c] = "unknown"
            elif c == "src":
                df[c] = "unknown"
            else:
                raise ValueError(f"Missing required column: {c}")
    # types
    df["mode"] = df["mode"].astype(str)
    return df[needed].copy()

def _dispersion(a: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    m = float(np.mean(a))
    s = float(np.std(a))
    return s/(abs(m)+1e-12)

def _unique_hparam_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["rho_target","leak","input_scale"]
    metrics = [c for c in ("MC","IPC","KR","GR") if c in df.columns]
    if not metrics:
        return df.copy()
    return (df.groupby(keys, as_index=False)[metrics]
              .mean()
              .sort_values(keys)
              .reset_index(drop=True))

def _compute_dispersion_table(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For each (mode, src, group_id), compute dispersion across hyper-params:
      - group_id = shuffle_id if shuffle_id != -1 else src
    """
    df = combined.copy()
    df["group_id"] = df["shuffle_id"].astype(str)
    df.loc[df["shuffle_id"] == -1, "group_id"] = df["src"].astype(str)

    # dedup repeated measurements within the same hyperparam triple
    keys = ["mode","src","group_id","rho_target","leak","input_scale"]
    metrics = [m for m in ("MC","IPC","KR","GR") if m in df.columns]
    df_agg = (df.groupby(keys, as_index=False)[metrics]
                .mean()
                .sort_values(keys)
                .reset_index(drop=True))

    rows = []
    for (mode, src, gid), grp in df_agg.groupby(["mode","src","group_id"]):
        for m in metrics:
            rows.append({
                "mode": mode,
                "src": src,
                "group_id": gid,
                "metric": m,
                "dispersion": _dispersion(grp[m].to_numpy()),
                "n_hparams": len(grp)
            })
    disp = pd.DataFrame(rows)
    return disp


# --------------------------- plots ---------------------------

def plot_overlaid_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    modes = sorted(disp["mode"].unique())

    # common bin edges per metric across all modes; y-axis normalized by count (fractions)
    for m in metrics:
        all_vals = disp[disp["metric"] == m]["dispersion"].to_numpy()
        if all_vals.size == 0:
            continue
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        edges = np.linspace(lo, hi, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.figure(figsize=(10, 6))
        for mode in modes:
            s = disp[(disp["metric"] == m) & (disp["mode"] == mode)]["dispersion"].to_numpy()
            if s.size == 0:
                continue
            counts, _ = np.histogram(s, bins=edges)
            frac = counts.astype(float) / max(len(s), 1)  # normalize by N so areas comparable
            plt.plot(centers, frac, drawstyle="steps-mid", linewidth=1.8, alpha=0.95,
                     label=f"{mode} (N={len(s)})")

        plt.title(f"Invariance dispersion by architecture — {m}")
        plt.xlabel(f"dispersion = std({m}) / mean({m}) across hyper-params")
        plt.ylabel("fraction (normalized by N)")
        plt.legend(frameon=False, fontsize=9, ncol=2)
        plt.tight_layout()
        out_fig = _safe_path(os.path.join(out_dir, f"all_arch_hist_{m}.png"))
        plt.savefig(out_fig, dpi=200)
        plt.close()
        print(f"[saved] {out_fig}")

def plot_mc_vs_gr_all_arch(combined: pd.DataFrame, out_dir: str, alpha: float):
    if not {"MC","GR","mode"}.issubset(combined.columns):
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(11, 6))
    # light grey background points for all runs
    plt.scatter(combined["GR"], combined["MC"], s=8, c="#bbbbbb", alpha=alpha, label="all")
    # emphasize CE-real and CE-shuffle if present
    for cname, color in [("CE-real", "#1f77b4"), ("CE-shuffle", "#d62728")]:
        sub = combined[combined["mode"] == cname]
        if not sub.empty:
            sub_u = _unique_hparam_rows(sub)
            plt.scatter(sub_u["GR"], sub_u["MC"], s=36, alpha=0.95, label=cname, c=color)
    plt.title("MC vs GR across all architectures")
    plt.xlabel("GR (effective rank of Δstate)")
    plt.ylabel("MC (linear memory capacity)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    out_fig = _safe_path(os.path.join(out_dir, "mc_vs_gr_all_arch.png"))
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved] {out_fig}")


# --------------------------- main ----------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.combined):
        raise FileNotFoundError(f"Combined CSV not found: {args.combined}")

    combined = pd.read_csv(args.combined)
    combined = _ensure_columns(combined)

    # Save a copy (non-destructive; versioned if exists)
    out_comb = _safe_path(os.path.join(args.out_dir, "combined.ALL.csv"))
    combined.to_csv(out_comb, index=False)
    print(f"[saved] {out_comb}  (rows={len(combined)})")

    # Compute and save dispersion table
    disp = _compute_dispersion_table(combined)
    out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    disp.to_csv(out_disp, index=False)
    print(f"[saved] {out_disp}  (rows={len(disp)})")

    # Plots
    plot_overlaid_arch_histograms(disp, args.out_dir, args.bins)
    plot_mc_vs_gr_all_arch(combined, args.out_dir, args.scatter_alpha)


if __name__ == "__main__":
    main()
