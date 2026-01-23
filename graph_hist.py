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
  experiment_full_merged/all_arch_hist_grid[_pN].png
  experiment_full_merged/mc_vs_gr_all_arch.png
"""

import os
import argparse
import numpy as np
import pandas as pd,itertools
import pingouin as pg
from scipy.stats import kruskal
from researchpy import difference_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util.graph_utils import _compute_dispersion_table

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
    return s/(abs(m)+1e-12) ## allows us to calcualte variance accross different models which have different scales

def _unique_hparam_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["rho_target","leak","input_scale"]
    metrics = [c for c in ("MC","IPC","KR","GR") if c in df.columns]
    if not metrics:
        return df.copy()
    return (df.groupby(keys, as_index=False)[metrics]
              .mean()
              .sort_values(keys)
              .reset_index(drop=True))




# --------------------------- plots ---------------------------

def plot_overlaid_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    modes = sorted(disp["mode"].unique())
    if not metrics:
        return
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    per_fig = 4  # 2x2 grid
    for start in range(0, len(metrics), per_fig):
        chunk = metrics[start:start + per_fig]
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), squeeze=False,
                                 sharex="col", sharey="row")
        flat_axes = axes.ravel()
        any_plotted = False
        col_lo = [np.inf, np.inf]
        col_hi = [-np.inf, -np.inf]
        row_y_max = [0.0, 0.0]
        legend_handles, legend_labels = None, None
        data_mask = [False] * len(flat_axes)

        for idx, m in enumerate(chunk):
            ax = flat_axes[idx]
            all_vals = disp[disp["metric"] == m]["dispersion"].to_numpy()
            if all_vals.size == 0:
                ax.axis("off")
                continue
            lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0
            edges = np.linspace(lo, hi, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            col = idx % 2
            row = idx // 2
            col_lo[col] = min(col_lo[col], lo)
            col_hi[col] = max(col_hi[col], hi)

            plotted = False
            for mode in modes:
                s = disp[(disp["metric"] == m) & (disp["mode"] == mode)]["dispersion"].to_numpy()
                if s.size == 0:
                    continue
                counts, _ = np.histogram(s, bins=edges)
                frac = counts.astype(float) / max(len(s), 1)  # normalize by N so areas comparable
                ax.plot(centers, frac, drawstyle="steps-mid", linewidth=1.8, alpha=0.95,
                        label=f"{mode} (N={len(s)})")
                plotted = True
                row_y_max[row] = max(row_y_max[row], float(np.max(frac)))

            if not plotted:
                ax.axis("off")
                continue

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.set_title(f"Invariance dispersion by architecture — {m}")
            if idx == 2 or idx ==3:
                ax.set_xlabel(f"coefficient of variation")
            if idx == 0 or idx ==2:
                ax.set_ylabel("fraction (normalized by N)")
            any_plotted = True
            data_mask[idx] = True

        for idx in range(len(chunk), len(flat_axes)):
            flat_axes[idx].axis("off")

        if not any_plotted:
            plt.close(fig)
            continue

        for col in range(2):
            if np.isfinite(col_lo[col]) and np.isfinite(col_hi[col]) and col_lo[col] != col_hi[col]:
                for row in range(2):
                    idx = row * 2 + col
                    ax = flat_axes[idx]
                    if data_mask[idx]:
                        ax.set_xlim(col_lo[col], col_hi[col])
        for row in range(2):
            if row_y_max[row] > 0:
                y_max = row_y_max[row] * 1.08
                for col in range(2):
                    idx = row * 2 + col
                    ax = flat_axes[idx]
                    if data_mask[idx]:
                        ax.set_ylim(0, y_max)

        if legend_handles:
            fig.legend(legend_handles, legend_labels, frameon=False, loc="upper center",
                       ncol=min(len(legend_handles), 3), bbox_to_anchor=(0.5, 1))

        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
        page = start // per_fig + 1
        suffix = "" if len(metrics) <= per_fig else f"_p{page}"
        out_fig = _safe_path(os.path.join(out_dir, f"all_arch_hist_grid{suffix}.png"))
        fig.savefig(out_fig, dpi=200)
        plt.close(fig)
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

def print_kruskal_wallis_tables(disp: pd.DataFrame):
    if kruskal is None:
        print("[warn] scipy is not available; skipping Kruskal-Wallis tables.")
        return
    metrics_present = set(disp["metric"].unique())
    metrics = [m for m in ["MC", "IPC", "KR", "GR"] if m in metrics_present]
    modes = sorted(disp["mode"].unique())
    if not metrics:
        print("[warn] none of MC/IPC/KR/GR present for testing.")
        return
    if not modes:
        print("[warn] no modes to test.")
        return
    print("\n=== Kruskal-Wallis tests (dispersion across modes within each metric) ===")
    for m in metrics:
        rows = []
        groups = []
        for mode in modes:
            vals = disp[(disp["metric"] == m) & (disp["mode"] == mode)]["dispersion"].dropna().to_numpy()
            if vals.size == 0:
                continue
            rows.append({"mode": mode,
                         'metric': m,
                         "N": len(vals),
                         "median": float(np.median(vals)),
                         "mean": float(np.mean(vals)),
                         "std": float(np.std(vals))})
            groups.append(vals)
        if len(groups) < 2:
            print(f"{m}: not enough non-empty modes for Kruskal-Wallis (need >=2).")
            continue
        H, p = kruskal(*groups)
        df = pd.DataFrame(rows)
        for a, b in itertools.combinations(df["mode"].unique(), 2):
                    row_a = df[df["mode"] == a].iloc[0]
                    row_b = df[df["mode"] == b].iloc[0]
                    sd_ref = row_a["std"] if row_a["std"] > 0 else np.nan
                    d_glass = (row_a["mean"] - row_b["mean"]) / sd_ref if np.isfinite(sd_ref) else np.nan
                    vals_a = disp[(disp["metric"] == m) & (disp["mode"] == a)]["dispersion"].dropna().to_numpy()
                    vals_b = disp[(disp["metric"] == m) & (disp["mode"] == b)]["dispersion"].dropna().to_numpy()
                    vals_all = disp[(disp["metric"] == m)]["dispersion"].dropna().to_numpy()
                    if a == "CE-real":
                        #print(f"  {a} vs {b} (Glass Δ, SD from {a}, {m}): {d_glass:.4g}")
                        tost = (pg.tost(vals_a,vals_b,np.abs(np.median(vals_all))*0.02,paired = False))
                        print(rf"{m} & {a} vs {b} & {tost['bound'].iloc[0]:.4g} & {tost['pval'].iloc[0]:.4g} \\")
        #print(f"\n{m}")
        #print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))  # one table per metric
        #print(f"H = {H:.4g}, p = {p:.4g}")
    print()


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
    disp = _compute_dispersion_table(combined,mode="cv")
    out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    disp.to_csv(out_disp, index=False)
    print(f"[saved] {out_disp}  (rows={len(disp)})")

    # Focus on CE-real, CE-shuffle, and CE-connshuff for plots/stats

    # Plots
    plot_overlaid_arch_histograms(disp, args.out_dir, args.bins)
    plot_mc_vs_gr_all_arch(combined, args.out_dir, args.scatter_alpha)
    print_kruskal_wallis_tables(disp)


if __name__ == "__main__":
    main()
