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
import pandas as pd
import itertools
import pingouin as pg
import warnings
from scipy.stats import kruskal
import matplotlib as mpl
import re

import matplotlib.pyplot as plt
if not os.environ.get("MPLBACKEND"):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg")
        

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
def plot_frac_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    if not metrics:
        return

    mode_order = [
        "sign_test0.0","sign_test0.1","sign_test0.2","sign_test0.3","sign_test0.4",
        "sign_test0.5","sign_test0.6","sign_test0.7","sign_test0.8","sign_test0.9","sign_test1.0",
    ]
    modes = [m for m in mode_order if m in set(disp["mode"].unique())]
    if not modes:
        return

    def mode_to_z(mode: str) -> float:
        m = re.search(r"sign_test([0-9]*\.?[0-9]+)", mode)
        return float(m.group(1)) if m else np.nan

    z_vals = np.array([mode_to_z(m) for m in modes], dtype=float)
    z_vals = z_vals[np.isfinite(z_vals)]
    zmin, zmax = float(np.min(z_vals)), float(np.max(z_vals))

    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=zmin, vmax=(zmax if zmax > zmin else zmin + 1.0))
    color_for_mode = {m: cmap(norm(mode_to_z(m))) for m in modes}

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    per_fig = 4  # 2x2
    for start in range(0, len(metrics), per_fig):
        chunk = metrics[start:start + per_fig]
        fig = plt.figure(figsize=(14, 12))
        axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

        any_plotted = False

        for idx, metric in enumerate(chunk):
            ax = axes[idx]
            all_vals = disp.loc[disp["metric"] == metric, "dispersion"].to_numpy()
            if all_vals.size == 0:
                ax.set_axis_off()
                continue

            lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0

            edges = np.linspace(lo, hi, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])

            plotted_this = False
            for mode in modes:
                s = disp[(disp["metric"] == metric) & (disp["mode"] == mode)]["dispersion"].to_numpy()
                if s.size == 0:
                    continue

                counts, _ = np.histogram(s, bins=edges)
                z = counts.astype(float) / max(len(s), 1)  # histogram fraction -> now z
                x = mode_to_z(mode)                          # DISCRETE mode value (0.1, 0.2, ...)
                if not np.isfinite(x):
                    continue

                y = centers                                # CV bin centers -> now y
                c = color_for_mode[mode]

                # 3D polyline at constant x
                ax.plot([x] * len(y), y, zs=z, zdir="z", linewidth=2.5, alpha=0.9, color=c)

                # Median CV marker line at constant x, spanning z
                #med = float(np.median(s))
                #z_max_local = float(np.max(z)) if len(z) else 0.0
                #ax.plot([x, x], [med, med], zs=[0.0, z_max_local], zdir="z",
                #        linewidth=1.2, alpha=0.5, color=c, linestyle="--")

                plotted_this = True

            if not plotted_this:
                ax.set_axis_off()
                continue

            any_plotted = True
            ax.set_title(f"{metric}")
            ax.set_xlabel("sign_test frac (x)")
            ax.set_ylabel("coefficient of variation (y)")
            ax.set_zlabel("fraction (hist, z)")

            # Make x discrete and readable
            xticks = [mode_to_z(m) for m in modes]
            xticks = sorted({float(v) for v in xticks if np.isfinite(v)})
            ax.set_xticks(xticks)

            ax.set_xlim(-0.05, 1.05)
            group_max = (
                disp.loc[disp["metric"] == metric]
                .groupby("mode")["dispersion"]
                .apply(lambda v: (
                    np.histogram(v.to_numpy(), bins=edges)[0].astype(float) / max(len(v), 1)
                ).max() if len(v) > 0 else 0.0)
            )

            ymax = float(group_max.max()) if len(group_max) > 0 else 0.0
            ax.set_zlim(0.0, 1.05 * ymax)
            ax.set_ylim(lo, hi)

            # Better default view for vertical sheets
            ax.view_init(elev=22, azim=-35)

        # turn off unused panels
        for j in range(len(chunk), 4):
            axes[j].set_axis_off()

        if not any_plotted:
            plt.close(fig)
            continue

        # colorbar keyed to z (fraction)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        #cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
        #cbar.set_label("sign_test frac (z)")

        fig.tight_layout()
        page = start // per_fig + 1
        suffix = "" if len(metrics) <= per_fig else f"_p{page}"
        out_fig = _safe_path(os.path.join(out_dir, f"all_arch_hist_grid_3d{suffix}.png"))
        fig.savefig(out_fig, dpi=300)
        plt.show()
        plt.close(fig)
        print(f"[saved] {out_fig}")






def plot_frac_cv_meanline(disp: pd.DataFrame, combined: pd.DataFrame, out_dir: str, show: bool = True):
    """
    Single 3D plot with one colored line per metric (MC, IPC, KR, GR):
      x = sign_test fraction (parsed from mode name)
      y = mean CV (dispersion) across groups
      z = mean performance across runs
    """
    os.makedirs(out_dir, exist_ok=True)

    # Expected metrics present in both tables.
    metric_cols = [m for m in ("MC", "IPC", "KR", "GR") if m in combined.columns]
    if not metric_cols:
        return
    metrics_disp = set(disp["metric"].unique())
    metrics = [m for m in metric_cols if m in metrics_disp]
    if not metrics:
        return

    # Fractions from mode names.
    def mode_to_frac(mode: str) -> float:
        m = re.search(r"sign_test([0-9]*\.?[0-9]+)", mode)
        return float(m.group(1)) if m else np.nan

    modes = sorted(
        {m for m in disp["mode"].unique() if re.match(r"sign_test", str(m))},
        key=lambda x: mode_to_frac(str(x)),
    )
    if not modes:
        return

    # Mean performance and mean CV lookups.
    comb_long = combined.melt(id_vars=["mode"], value_vars=metrics, var_name="metric", value_name="value")
    mean_perf = comb_long.groupby(["mode", "metric"])["value"].mean()
    mean_cv = disp.groupby(["mode", "metric"])["dispersion"].mean()

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    colors = mpl.colormaps["tab10"]
    plotted_any = False
    for idx, metric in enumerate(metrics):
        xs, ys, zs = [], [], []
        for mode in modes:
            frac = mode_to_frac(mode)
            y = mean_cv.get((mode, metric), np.nan)
            z = mean_perf.get((mode, metric), np.nan)
            if not (np.isfinite(frac) and np.isfinite(y) and np.isfinite(z)):
                continue
            xs.append(frac)
            ys.append(y)
            zs.append(z)
        if not xs:
            continue
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        zs = np.asarray(zs)[order]
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("sign_test frac (x)")
    ax.set_ylabel("mean CV (y)")
    ax.set_zlabel("mean performance (z)")
    ax.set_title("Mean performance vs CV vs sign_test fraction")
    ax.set_xlim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()

    out_fig = _safe_path(os.path.join(out_dir, "meanpoint_frac_cv_lines.png"))
    fig.savefig(out_fig, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_fig}")
    print(f"[saved] {out_fig}")
def plot_overlaid_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    # Consistent ordering/colors to make panels easier to compare.
    mode_order = [
        "real",
        "cel+randN",
        "er+randN",
        "ws_p0.1+randN",
        "celW+connShuf",
        "local_sign",
        "shuffle",
        "cel_sample",
        "conn_shuf_only",
        "local_sign+flat",
        "local_sign+sample",
        "local_sign+binary",
        "global_sign_pres"

    ]
    modes = [m for m in mode_order if m in set(disp["mode"].unique())]
    color_map = {
        "real": "#32a2f2",
        "cel+randN": "#127212",
        "er+randN": "#6eb775",
        "ws_p0.1+randN": "#40FF00",
        "celW+connShuf": "#ff0e0e",
        "local_sign": "#027979",
        "shuffle": "#FF5100",
        "cel_sample": "#ff6565",
        "conn_shuf_only": "#ff9752",
        "local_sign+flat":  "#002CF1",
        "local_sign+sample": "#AF7DF5",
        "local_sign+binary": "#8ADEF3",
        "global_sign_pres" : "#1C373D",

    }
    if not metrics:
        return
    plt.rcParams.update({
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 20,
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
                color = color_map.get(mode, None)
                name_remap = {
                    "real": "C. elegans",
                    "cel+randN": "C. elegans + N(0,1)",
                    "er+randN": "ER + N(0,1)",
                    "celW+connShuf": "Conn+wt shuffle",
                    "shuffle": "Weight shuffle",
                    "conn_shuf_only": "Connection shuffle",
                    "local_sign": "Local Sign Preserved + N(0,1)",
                    "ws_p0.1+randN": "WS p=0.1 + N(0,1)",
                    "cel_sample": "Sampled weights",
                    "local_sign+flat": "Local Sign + U(0,1)",
                    "local_sign+sample": "Local Sign + Sampled",
                    "local_sign+binary": "Local Sign + wt +1,-1",
                    "global_sign_pres" : "Global sign",



                }
                mode = name_remap[mode]
                ax.plot(
                    centers,
                    frac,
                    drawstyle="steps-mid",
                    linewidth=3.0,
                    alpha=0.7,
                    label=f"{mode}",
                    color=color,
                )
                # Median marker to help compare shifts without cluttering the plot.
                med = float(np.median(s))
                ax.axvline(med, color=color, alpha=0.18, linewidth=1.2, linestyle="--")
                plotted = True
                row_y_max[row] = max(row_y_max[row], float(np.max(frac)))

            if not plotted:
                ax.axis("off")
                continue

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.set_title(f"Invariance dispersion by architecture — {m}")
            if idx == 2 or idx ==3:
                ax.set_xlabel("coefficient of variation")
            if idx == 0 or idx ==2:
                ax.set_ylabel("fraction (normalized by N)")
            ax.grid(True, which="both", axis="both", alpha=0.18, linestyle=":")
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
            fig.legend(
                legend_handles,
                legend_labels,
                frameon=False,
                loc="upper center",
                ncol=3,
                bbox_to_anchor=(0.5, 1.02),
                columnspacing=1.2,
                handlelength=2.0,
                handletextpad=0.6,
                borderaxespad=0.8,
            )

        fig.tight_layout(rect=(0.00, 0.00, 0.96, 0.85))
        page = start // per_fig + 1
        suffix = "" if len(metrics) <= per_fig else f"_p{page}"
        out_fig = _safe_path(os.path.join(out_dir, f"all_arch_hist_grid{suffix}.png"))
        fig.savefig(out_fig, dpi=300)
        plt.close(fig)
        print(f"[saved] {out_fig}")

def plot_mc_vs_gr_all_arch(combined: pd.DataFrame, out_dir: str, alpha: float):
    if not {"MC","GR","mode"}.issubset(combined.columns):
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(11, 6))
    # light grey background points for all runs
    plt.scatter(combined["GR"], combined["IPC"], s=8, c="#bbbbbb", alpha=alpha, label="all")
    # emphasize CE-real and CE-shuffle if present
    for cname, color in [("local_sign+binary", "#1f77b4"), ("real", "#d62728")]:
        sub = combined[combined["mode"] == cname]
        if not sub.empty:
            sub_u = _unique_hparam_rows(sub)
            plt.scatter(sub_u["GR"], sub_u["MC"], s=36, alpha=0.5, label=cname, c=color)
    plt.title("MC vs GR across all architectures")
    plt.xlabel("GR (effective rank of Δstate)")
    plt.ylabel("MC (linear memory capacity)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    out_fig = _safe_path(os.path.join(out_dir, "mc_vs_gr_all_arch.png"))
    plt.savefig(out_fig, dpi=300)
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
                    if a == "real" or b =="real":
                        # pg.tost can emit overflow RuntimeWarnings for extreme t values; suppress locally
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=RuntimeWarning, module="pingouin")
                            tost = (pg.tost(vals_a, vals_b, np.abs(np.median(vals_all)) * 0.05, paired=False))
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
    #combined.to_csv(out_comb, index=False)
    print(f"[saved] {out_comb}  (rows={len(combined)})")

    # Compute and save dispersion table
    disp = _compute_dispersion_table(combined,mode="cv")
    out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    #disp.to_csv(out_disp, index=False)

    # Plots
    #plot_frac_arch_histograms(disp, args.out_dir, args.bins)
    #plot_frac_cv_meanline(disp, combined, args.out_dir)
    plot_overlaid_arch_histograms(disp, args.out_dir, args.bins)
    #plot_mc_vs_gr_all_arch(combined, args.out_dir, args.scatter_alpha)
    #print_kruskal_wallis_tables(disp)


if __name__ == "__main__":
    main()
