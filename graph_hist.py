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
import csv
import numpy as np
import pandas as pd
import itertools
import pingouin as pg
import warnings
from scipy.stats import kruskal
import matplotlib as mpl
import re

if not os.environ.get("MPLBACKEND"):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        mpl.use("Agg")

import matplotlib.pyplot as plt

from util.graph_utils import (
    _compute_dispersion_table,
    _compute_mean_table,
    _assign_group_ids,
    _aggregate_over_hparams,
)

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
    ap.add_argument(
        "--model",
        default="",
        help="Optional exact mode/model name filter (e.g., 'weight_test10.0') for rho/CV/performance plot.",
    )
    ap.add_argument(
        "--compare-mean-a",
        default="",
        help="Optional path to first comparison CSV (mean_by_group.ALL.csv or dispersion_by_group.ALL.csv).",
    )
    ap.add_argument(
        "--compare-mean-b",
        default="",
        help="Optional path to second comparison CSV (mean_by_group.ALL.csv or dispersion_by_group.ALL.csv).",
    )
    ap.add_argument(
        "--compare-label-a",
        default="full",
        help="Column label for --compare-mean-a output columns.",
    )
    ap.add_argument(
        "--compare-label-b",
        default="frac025",
        help="Column label for --compare-mean-b output columns.",
    )
    ap.add_argument(
        "--compare-mean-out",
        default="",
        help="Optional output CSV path for mode/metric mean comparison table.",
    )
    ap.add_argument(
        "--compare-plot-out",
        default="",
        help="Optional output PNG path for all-modes mean-difference plot.",
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Run mean-table comparison only, then exit.",
    )
    ap.add_argument(
        "--compare-value-col",
        default="auto",
        help="Column to compare: 'auto', 'mean', or 'dispersion' (use 'dispersion' for CV means).",
    )
    ap.add_argument(
        "--compare-metric",
        default="",
        help="Optional metric filter for comparison plot (e.g., 'cosine_similarity' or 'covariance').",
    )
    ap.add_argument(
        "--compare-tost-preservation",
        action="store_true",
        help="Print how many baseline-significant pairwise TOST comparisons remain significant in B.",
    )
    ap.add_argument(
        "--compare-tost-alpha",
        type=float,
        default=0.05,
        help="Alpha threshold for TOST significance preservation reporting.",
    )
    ap.add_argument(
        "--compare-tost-bound-frac",
        type=float,
        default=0.05,
        help="Equivalence bound as fraction of |baseline metric median|, reused for both datasets.",
    )
    ap.add_argument(
        "--compare-tost-out",
        default="",
        help="Optional CSV path for per-pair TOST preservation details.",
    )
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


def _read_combined_csv(path: str) -> pd.DataFrame:
    """Read combined CSV and trim malformed extra columns per row if needed."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError as exc:
        print(f"[warn] {exc}")
        print(f"[warn] Retrying by trimming malformed extra column(s) from {path}.")

        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise

            expected_fields = len(header)
            rows = []
            trimmed_rows = 0
            padded_rows = 0
            first_trimmed_line = None
            first_padded_line = None

            for line_no, row in enumerate(reader, start=2):
                n_fields = len(row)
                if n_fields > expected_fields:
                    if first_trimmed_line is None:
                        first_trimmed_line = line_no
                    rows.append(row[:expected_fields])
                    trimmed_rows += 1
                elif n_fields < expected_fields:
                    if first_padded_line is None:
                        first_padded_line = line_no
                    rows.append(row + [""] * (expected_fields - n_fields))
                    padded_rows += 1
                else:
                    rows.append(row)

        if trimmed_rows == 0 and padded_rows == 0:
            raise

        print(
            f"[warn] Trimmed extra trailing column(s) on {trimmed_rows} row(s). "
            f"First trimmed line: {first_trimmed_line}."
        )
        if padded_rows:
            print(
                f"[warn] Padded missing trailing column(s) on {padded_rows} row(s). "
                f"First padded line: {first_padded_line}."
            )
        df = pd.DataFrame(rows, columns=header)
        numeric_cols = [
            "shuffle_id",
            "rho_target",
            "leak",
            "input_scale",
            "MC",
            "IPC",
            "KR",
            "GR",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


def _read_compare_table(path: str, value_col: str = "auto"):
    df = pd.read_csv(path)
    required = {"mode", "metric"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Comparison table {path} is missing required column(s): {sorted(missing)}")

    col = (value_col or "auto").strip().lower()
    if col == "auto":
        if "mean" in df.columns:
            col = "mean"
        elif "dispersion" in df.columns:
            col = "dispersion"
        else:
            raise ValueError(
                f"Could not infer comparison value column for {path}. "
                "Expected one of: 'mean', 'dispersion'."
            )
    elif col not in ("mean", "dispersion"):
        raise ValueError("--compare-value-col must be one of: auto, mean, dispersion")

    if col not in df.columns:
        raise ValueError(
            f"Requested comparison column '{col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[col] = pd.to_numeric(df[col], errors="coerce")
    if "n_hparams" in df.columns:
        df["n_hparams"] = pd.to_numeric(df["n_hparams"], errors="coerce")
    return df, col


def _mode_metric_means(comp_tbl: pd.DataFrame, value_col: str = "mean") -> pd.DataFrame:
    cols = ["mode", "metric", value_col]
    if "n_hparams" in comp_tbl.columns:
        cols.append("n_hparams")
    tbl = comp_tbl[cols].copy()
    if "n_hparams" not in tbl.columns:
        tbl["n_hparams"] = 1.0

    tbl = tbl.dropna(subset=["mode", "metric", value_col])
    tbl.loc[tbl["n_hparams"] <= 0, "n_hparams"] = np.nan
    tbl["weighted_sum"] = tbl[value_col] * tbl["n_hparams"]

    grp = tbl.groupby(["mode", "metric"], as_index=False, dropna=False)
    agg = grp.agg(
        weighted_sum=("weighted_sum", "sum"),
        weight_sum=("n_hparams", "sum"),
        simple_mean=(value_col, "mean"),
        n_groups=(value_col, "size"),
    )
    agg["mode_metric_mean"] = agg["weighted_sum"] / agg["weight_sum"]
    agg["mode_metric_mean"] = agg["mode_metric_mean"].where(
        np.isfinite(agg["mode_metric_mean"]),
        agg["simple_mean"],
    )
    return agg[["mode", "metric", "mode_metric_mean", "n_groups"]]


def compare_mode_metric_means(
    mean_csv_a: str,
    mean_csv_b: str,
    out_csv: str,
    label_a: str = "full",
    label_b: str = "frac025",
    value_col: str = "auto",
) -> pd.DataFrame:
    a_raw, col_a = _read_compare_table(mean_csv_a, value_col=value_col)
    b_raw, col_b = _read_compare_table(mean_csv_b, value_col=value_col)
    if col_a != col_b:
        raise ValueError(
            f"Comparison value columns do not match: A uses '{col_a}', B uses '{col_b}'. "
            "Set --compare-value-col explicitly."
        )
    print(f"[compare] using value column: {col_a}")

    a = _mode_metric_means(a_raw, value_col=col_a).rename(
        columns={
            "mode_metric_mean": f"mean_{label_a}",
            "n_groups": f"n_groups_{label_a}",
        }
    )
    b = _mode_metric_means(b_raw, value_col=col_b).rename(
        columns={
            "mode_metric_mean": f"mean_{label_b}",
            "n_groups": f"n_groups_{label_b}",
        }
    )

    out = a.merge(b, on=["mode", "metric"], how="outer")
    out["delta"] = out[f"mean_{label_b}"] - out[f"mean_{label_a}"]
    denom = out[f"mean_{label_a}"].abs()
    out["pct_delta"] = np.where(denom > 0, 100.0 * out["delta"] / denom, np.nan)
    out["drop_from_a_to_b"] = out[f"mean_{label_a}"] - out[f"mean_{label_b}"]
    out["pct_drop_from_a_to_b"] = np.where(
        denom > 0,
        100.0 * out["drop_from_a_to_b"] / denom,
        np.nan,
    )
    out = out.sort_values(["metric", "mode"], kind="stable").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv} (rows={len(out)})")

    for metric in out["metric"].dropna().unique():
        sub = out[(out["metric"] == metric) & np.isfinite(out["delta"])].copy()
        if sub.empty:
            continue
        top = sub.iloc[sub["delta"].abs().argmax()]
        print(
            f"[compare] {metric}: max |delta| mode={top['mode']} "
            f"delta={top['delta']:.6g} ({label_b} - {label_a})"
        )
    return out


def _normalize_compare_metrics(spec: str):
    if not spec:
        return []
    aliases = {
        "cov": "cosine_similarity",
        "covariance": "cosine_similarity",
        "covariance_mean": "cosine_similarity",
        "cosine": "cosine_similarity",
    }
    out = []
    for tok in str(spec).split(","):
        key = tok.strip()
        if not key:
            continue
        out.append(aliases.get(key.lower(), key))
    return out


def plot_mode_self_drop_comparison(
    comp_tbl: pd.DataFrame,
    out_png: str,
    label_a: str = "full",
    label_b: str = "frac025",
):
    req = {"mode", "metric"}
    if not req.issubset(comp_tbl.columns):
        print(f"[warn] comparison table missing columns for plotting: {sorted(req)}")
        return

    plot_value_col = "pct_drop_from_a_to_b" if "pct_drop_from_a_to_b" in comp_tbl.columns else "drop_from_a_to_b"
    dat = comp_tbl.dropna(subset=["mode", "metric", plot_value_col]).copy()
    if dat.empty:
        print("[warn] no overlapping mode/metric rows to plot self-drop comparison.")
        return

    metric_order = ["MC", "IPC", "KR", "GR", "cosine_similarity", "wt_mean"]
    metrics_present = list(dat["metric"].dropna().unique())
    metrics = [m for m in metric_order if m in metrics_present] + [m for m in metrics_present if m not in metric_order]
    pivot = dat.pivot_table(
        index="mode",
        columns="metric",
        values=plot_value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=metrics)
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

    vals = pivot.to_numpy(dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    vmax = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 1.0
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    fig_w = max(8.0, 1.8 * len(metrics) + 3.0)
    fig_h = max(5.0, 0.45 * len(pivot.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = mpl.colormaps["coolwarm"].copy()
    cmap.set_bad(color="#e6e6e6")
    masked = np.ma.masked_invalid(vals)
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Mode")
    if plot_value_col == "pct_drop_from_a_to_b":
        ax.set_title(f"Per-mode self drop heatmap (%: {label_a} vs {label_b})")
    else:
        ax.set_title(f"Per-mode self drop heatmap ({label_a} - {label_b})")

    # Annotate cell values for fast lookup.
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="#666666")
                continue
            text_color = "white" if abs(v) > (0.55 * vmax) else "black"
            if plot_value_col == "pct_drop_from_a_to_b":
                label = f"{v:.1f}%"
            else:
                label = f"{v:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if plot_value_col == "pct_drop_from_a_to_b":
        cbar.set_label(f"Percent drop (%): ({label_a} - {label_b}) / |{label_a}|")
    else:
        cbar.set_label(f"Drop ({label_a} - {label_b})")
    fig.suptitle(f"Positive means {label_b} is lower than {label_a}", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


def _metric_bounds_from_reference(ref_tbl: pd.DataFrame, value_col: str, bound_frac: float):
    bounds = {}
    for metric in sorted(ref_tbl["metric"].dropna().unique()):
        vals = ref_tbl.loc[ref_tbl["metric"] == metric, value_col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        med = float(np.nanmedian(vals))
        bounds[metric] = abs(med) * float(bound_frac)
    return bounds


def _pairwise_tost_results(
    tbl: pd.DataFrame,
    value_col: str,
    bounds_by_metric: dict,
    alpha: float = 0.05,
):
    out_rows = []
    metrics = sorted(tbl["metric"].dropna().unique())
    for metric in metrics:
        sub_metric = tbl[tbl["metric"] == metric]
        modes = sorted(sub_metric["mode"].dropna().unique())
        bound = float(bounds_by_metric.get(metric, np.nan))
        if not np.isfinite(bound):
            continue
        for mode_a, mode_b in itertools.combinations(modes, 2):
            vals_a = sub_metric.loc[sub_metric["mode"] == mode_a, value_col].dropna().to_numpy(dtype=float)
            vals_b = sub_metric.loc[sub_metric["mode"] == mode_b, value_col].dropna().to_numpy(dtype=float)
            if vals_a.size == 0 or vals_b.size == 0:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="pingouin")
                tost = pg.tost(vals_a, vals_b, bound, paired=False)
            pval = float(tost["pval"].iloc[0])
            sig = bool(np.isfinite(pval) and (pval < alpha))
            out_rows.append(
                {
                    "metric": metric,
                    "mode_a": mode_a,
                    "mode_b": mode_b,
                    "bound_ref": bound,
                    "pval": pval,
                    "sig": sig,
                }
            )
    return pd.DataFrame(out_rows)


def print_tost_preservation_summary(
    ref_tbl: pd.DataFrame,
    new_tbl: pd.DataFrame,
    value_col: str,
    label_ref: str,
    label_new: str,
    alpha: float = 0.05,
    bound_frac: float = 0.05,
    metrics_filter=None,
    out_csv: str = "",
):
    ref = ref_tbl.copy()
    new = new_tbl.copy()
    ref[value_col] = pd.to_numeric(ref[value_col], errors="coerce")
    new[value_col] = pd.to_numeric(new[value_col], errors="coerce")
    ref = ref.dropna(subset=["mode", "metric", value_col])
    new = new.dropna(subset=["mode", "metric", value_col])

    common_modes = sorted(set(ref["mode"].unique()) & set(new["mode"].unique()))
    ref = ref[ref["mode"].isin(common_modes)].copy()
    new = new[new["mode"].isin(common_modes)].copy()
    common_metrics = sorted(set(ref["metric"].unique()) & set(new["metric"].unique()))
    if metrics_filter:
        common_metrics = [m for m in common_metrics if m in set(metrics_filter)]
    ref = ref[ref["metric"].isin(common_metrics)].copy()
    new = new[new["metric"].isin(common_metrics)].copy()

    bounds = _metric_bounds_from_reference(ref, value_col=value_col, bound_frac=bound_frac)
    ref_res = _pairwise_tost_results(ref, value_col=value_col, bounds_by_metric=bounds, alpha=alpha)
    new_res = _pairwise_tost_results(new, value_col=value_col, bounds_by_metric=bounds, alpha=alpha)

    if ref_res.empty or new_res.empty:
        print("[tost] no pairwise tests available for preservation summary.")
        return

    merged = ref_res.merge(
        new_res,
        on=["metric", "mode_a", "mode_b", "bound_ref"],
        how="outer",
        suffixes=(f"_{label_ref}", f"_{label_new}"),
    )
    sig_ref_col = f"sig_{label_ref}"
    sig_new_col = f"sig_{label_new}"
    p_ref_col = f"pval_{label_ref}"
    p_new_col = f"pval_{label_new}"
    merged[sig_ref_col] = merged[sig_ref_col].fillna(False).astype(bool)
    merged[sig_new_col] = merged[sig_new_col].fillna(False).astype(bool)
    merged["preserved_sig"] = merged[sig_ref_col] & merged[sig_new_col]

    base_sig = int(merged[sig_ref_col].sum())
    base_sig_testable = int(merged.loc[merged[p_new_col].notna(), sig_ref_col].sum())
    preserved = int(merged["preserved_sig"].sum())
    new_sig = int(merged[sig_new_col].sum())
    pct_all = (100.0 * preserved / base_sig) if base_sig > 0 else np.nan
    pct_testable = (100.0 * preserved / base_sig_testable) if base_sig_testable > 0 else np.nan

    print("\n=== TOST Significance Preservation (pairwise mode comparisons) ===")
    print(
        f"[tost] baseline significant in {label_ref}: {base_sig} "
        f"(alpha={alpha:.3g}, bound_frac={bound_frac:.3g})"
    )
    print(f"[tost] significant in {label_new}: {new_sig}")
    print(
        f"[tost] preserved baseline significances in {label_new}: "
        f"{preserved}/{base_sig} ({pct_all:.1f}%)"
        if np.isfinite(pct_all)
        else f"[tost] preserved baseline significances in {label_new}: {preserved}/{base_sig}"
    )
    print(
        f"[tost] testable-preservation (common tested pairs): "
        f"{preserved}/{base_sig_testable} ({pct_testable:.1f}%)"
        if np.isfinite(pct_testable)
        else f"[tost] testable-preservation (common tested pairs): {preserved}/{base_sig_testable}"
    )

    for metric in common_metrics:
        sub = merged[merged["metric"] == metric]
        b = int(sub[sig_ref_col].sum())
        p = int(sub["preserved_sig"].sum())
        pct = (100.0 * p / b) if b > 0 else np.nan
        if np.isfinite(pct):
            print(f"[tost] {metric}: preserved {p}/{b} ({pct:.1f}%)")
        else:
            print(f"[tost] {metric}: preserved {p}/{b}")
    print()



    # Show baseline-significant comparisons that are not preserved in new run.
    not_preserved = merged[(merged[sig_ref_col]) & (~merged[sig_new_col])].copy()
    if not not_preserved.empty:
        not_preserved = not_preserved.sort_values(["metric", "mode_a", "mode_b"], kind="stable")
        print("[tost] not preserved comparisons (baseline significant, new not significant):")
        for _, row in not_preserved.iterrows():
            p_ref = row.get(p_ref_col, np.nan)
            p_new = row.get(p_new_col, np.nan)
            if np.isfinite(p_new):
                print(
                    f"[tost] {row['metric']}: {row['mode_a']} vs {row['mode_b']} "
                    f"(p_{label_ref}={p_ref:.4g}, p_{label_new}={p_new:.4g}, bound={row['bound_ref']:.4g})"
                )
            else:
                print(
                    f"[tost] {row['metric']}: {row['mode_a']} vs {row['mode_b']} "
                    f"(p_{label_ref}={p_ref:.4g}, p_{label_new}=NA, bound={row['bound_ref']:.4g})"
                )
        print()


def _save_publication_figure(fig, out_path: str, dpi: int = 600):
    """Save figure using paper-friendly defaults."""
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="white",
    )


def _tight_layout_quiet(fig):
    """Apply tight layout while suppressing benign layout warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)
        fig.tight_layout()


def _style_3d_axis(ax, tick_labelsize: int = 11, tick_pad: int = 2):
    """Apply a cleaner, publication-friendly 3D style."""
    ax.grid(True, which="major", linestyle=":", alpha=0.28)
    ax.tick_params(axis="x", which="major", labelsize=tick_labelsize, pad=0)
    ax.tick_params(axis="y", which="major", labelsize=tick_labelsize, pad=-2)
    ax.tick_params(axis="z", which="major", labelsize=tick_labelsize, pad=2)
    y_formatter = ax.yaxis.get_major_formatter()
    if isinstance(y_formatter, mpl.ticker.ScalarFormatter):
        y_formatter.set_useOffset(False)
        y_formatter.set_scientific(False)
    ax.yaxis.get_offset_text().set_visible(False)

    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    except Exception:
        pass


def _format_wt_mean_axis_as_offset(ax, wt_vals_scaled):
    """
    Compact Wt-mean tick labels by subtracting a shared 2-decimal baseline.
    Axis values are already scaled by 10**z_power.
    """
    if not wt_vals_scaled:
        return None
    vals = np.asarray(wt_vals_scaled, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    base = np.floor(np.nanmin(vals) * 100.0) / 100.0
    # Show only the fine variation around the shared base; values are in micro-units.
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _pos: f"{(v - base) * 1000.0:.1f}")
    )
    ax.yaxis.get_offset_text().set_visible(False)
    return base


def _capture_figure_rgba(fig):
    """Render a figure and return an RGBA image array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()


def _trim_white_border(img: np.ndarray, tol: int = 250, pad: int = 20) -> np.ndarray:
    """Trim mostly-white border from an RGBA image."""
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    rgb = img[..., :3]
    alpha = img[..., 3] if img.shape[2] >= 4 else np.full(img.shape[:2], 255, dtype=np.uint8)
    mask = (np.any(rgb < tol, axis=2)) & (alpha > 0)

    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    if ys.size == 0 or xs.size == 0:
        return img

    y0 = max(int(ys[0]) - pad, 0)
    y1 = min(int(ys[-1]) + pad + 1, img.shape[0])
    x0 = max(int(xs[0]) - pad, 0)
    x1 = min(int(xs[-1]) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def _pad_height_center(img: np.ndarray, target_h: int) -> np.ndarray:
    """Pad image to target height with white background, centered vertically."""
    h, w = img.shape[:2]
    if h >= target_h:
        return img
    out = np.full((target_h, w, 4), 255, dtype=np.uint8)
    top = (target_h - h) // 2
    out[top:top + h, :w] = img
    return out


def _save_3d_front_back(
    fig,
    axes,
    out_png: str,
    front_view=(22, -35),
    back_view=(22, 145),
    dpi: int = 600,
    tick_labelsize: int = 11,
    tick_pad: int = 2,
):
    """Save one publication PNG with front/back views side-by-side."""
    if isinstance(axes, (list, tuple, np.ndarray)):
        axes_list = [ax for ax in axes if ax is not None]
    else:
        axes_list = [axes] if axes is not None else []

    for ax in axes_list:
        _style_3d_axis(ax, tick_labelsize=tick_labelsize, tick_pad=tick_pad)

    front_png = _safe_path(out_png)
    front_root, front_ext = os.path.splitext(front_png)
    if not front_ext:
        front_ext = ".png"
        front_png = f"{front_png}{front_ext}"
        front_root, _ = os.path.splitext(front_png)

    original_views = []
    for ax in axes_list:
        original_views.append((ax.elev, ax.azim))

    for ax in axes_list:
        ax.view_init(elev=front_view[0], azim=front_view[1])
    img_front = _trim_white_border(_capture_figure_rgba(fig))

    for ax in axes_list:
        ax.view_init(elev=back_view[0], azim=back_view[1])
    img_back = _trim_white_border(_capture_figure_rgba(fig))

    target_h = max(img_front.shape[0], img_back.shape[0])
    img_front = _pad_height_center(img_front, target_h)
    img_back = _pad_height_center(img_back, target_h)

    gap_px = 0
    canvas_w = img_front.shape[1] + gap_px + img_back.shape[1]
    canvas = np.full((target_h, canvas_w, 4), 255, dtype=np.uint8)
    canvas[:, :img_front.shape[1]] = img_front
    canvas[:, img_front.shape[1] + gap_px:] = img_back

    panel_fig = plt.figure(
        figsize=(canvas.shape[1] / float(dpi), canvas.shape[0] / float(dpi)),
        dpi=dpi,
    )
    ax_panel = panel_fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_panel.imshow(canvas)
    ax_panel.axis("off")

    _save_publication_figure(panel_fig, front_png, dpi=dpi)
    plt.close(panel_fig)

    for ax, (elev, azim) in zip(axes_list, original_views):
        ax.view_init(elev=elev, azim=azim)

    return [front_png]

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["mode","shuffle_id","rho_target","leak","input_scale","MC","IPC","KR","GR","src"]
    df = df.copy()
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
    extras = [c for c in df.columns if c not in needed]
    return df[needed + extras].copy()

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
    #mode_order = [
    #    "weight_test0.0","weight_test1.0","weight_test5.0","weight_test10.0","weight_test100.0",
    #    "weight_test1000.0", "weight_test10000.0",
    #]
    modes = [m for m in mode_order if m in set(disp["mode"].unique())]
    if not modes:
        return

    def mode_to_value(mode: str) -> float:
        """Extract numeric value from a mode name.

        Supports both legacy "sign_testX" and current "weight_testX" patterns.
        Falls back to the first numeric token if no known prefix matches.
        Returns NaN when no numeric component is present.
        """
        mode_str = str(mode)
        for prefix in ("sign_test", "weight_test"):
            m = re.search(rf"{prefix}([0-9]*\.?[0-9]+)", mode_str)
            if m:
                return float(m.group(1))
        m = re.search(r"([0-9]*\.?[0-9]+)", mode_str)
        return float(m.group(1)) if m else np.nan

    mode_values = {m: mode_to_value(m) for m in modes}
    z_vals = np.array([v for v in mode_values.values() if np.isfinite(v)], dtype=float)

    # If nothing numeric was found, fall back to ordinal positions so the
    # plotting code remains robust instead of crashing.
    if z_vals.size == 0:
        mode_values = {m: float(i) for i, m in enumerate(modes)}
        z_vals = np.array(list(mode_values.values()), dtype=float)

    zmin, zmax = float(np.min(z_vals)), float(np.max(z_vals))
    if zmax == zmin:
        zmax = zmin + 1.0

    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)
    color_for_mode = {m: cmap(norm(mode_values[m])) for m in modes}

    if all(str(m).startswith("sign_test") for m in modes):
        x_label = "sign_test frac (x)"
    elif all(str(m).startswith("weight_test") for m in modes):
        x_label = "weight_test value (x)"
    else:
        x_label = "mode value (x)"

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
                x = mode_values.get(mode, np.nan)          # numeric mode value
                if not np.isfinite(x):
                    continue

                y = centers                                # CV bin centers -> now y
                c = color_for_mode[mode]

                # 3D polyline at constant x
                ax.plot([x] * len(y), y, zs=z, zdir="z", linewidth=2.5, alpha=0.9, color=c)

                # Median CV marker line at constant x, spanning z
                #med = float(np.median(s))
                #z_max_local = float(np.max) if len else 0.0
                #ax.plot([x, x], [med, med], zs=[0.0, z_max_local], zdir="z",
                #        linewidth=1.2, alpha=0.5, color=c, linestyle="--")

                plotted_this = True

            if not plotted_this:
                ax.set_axis_off()
                continue

            any_plotted = True
            ax.set_title(f"{metric}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("coefficient of variation ")
            ax.set_zlabel("fraction (hist, z)")

            # Make x discrete and readable
            xticks = [mode_values[m] for m in modes if np.isfinite(mode_values[m])]
            xticks = sorted({float(v) for v in xticks})
            if xticks:
                ax.set_xticks(xticks)
                span = max(xticks) - min(xticks)
                pad = 0.05 * span if span > 0 else 1.0
                ax.set_xlim(min(xticks) - pad, max(xticks) + pad)
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
        #cbar.set_label("sign_test frac ")

        fig.tight_layout()
        page = start // per_fig + 1
        suffix = "" if len(metrics) <= per_fig else f"_p{page}"
        out_fig = os.path.join(out_dir, f"all_arch_hist_grid_3d{suffix}.png")
        saved_paths = _save_3d_front_back(
            fig,
            axes,
            out_fig,
            front_view=(22, -35),
            back_view=(22, 145),
            dpi=600,
        )
        plt.show()
        plt.close(fig)
        for out_path in saved_paths:
            print(f"[saved] {out_path}")






def plot_frac_cv_meanline(disp: pd.DataFrame, combined: pd.DataFrame, out_dir: str, show: bool = True):
    """
    Single 3D plot with one colored line per metric (MC, IPC, KR, GR):
      x = mode value (parsed numeric component)
      y = mean CV (dispersion) across groups
      z = mean performance across runs
    """
    os.makedirs(out_dir, exist_ok=True)

    # Expected metrics present in both tables.
    metric_cols = [m for m in ("MC", "IPC", "KR", "GR") if m in combined.columns]
    if not metric_cols:
        print("[warn] plot_frac_cv_meanline: no MC/IPC/KR/GR columns found.")
        return
    metrics_disp = set(disp["metric"].unique())
    metrics = [m for m in metric_cols if m in metrics_disp]
    if not metrics:
        print("[warn] plot_frac_cv_meanline: dispersion table missing MC/IPC/KR/GR metrics.")
        return

    def mode_to_value(mode: str) -> float:
        mode_str = str(mode)
        for prefix in ("sign_test", "weight_test"):
            m = re.search(rf"{prefix}([0-9]*\.?[0-9]+)", mode_str)
            if m:
                return float(m.group(1))
        m = re.search(r"([0-9]*\.?[0-9]+)", mode_str)
        return float(m.group(1)) if m else np.nan

    # Keep only modes we can assign a numeric position to; sort by that value.
    mode_values = {m: mode_to_value(m) for m in disp["mode"].unique()}
    modes = [m for m, v in mode_values.items() if np.isfinite(v)]
    modes = sorted(modes, key=lambda m: mode_values[m])
    if not modes:
        print("[warn] plot_frac_cv_meanline: no modes with numeric value; skipping.")
        return

    # Mean performance and mean CV lookups.
    comb_long = combined.melt(id_vars=["mode"], value_vars=metrics, var_name="metric", value_name="value")
    mean_perf = comb_long.groupby(["mode", "metric"])["value"].mean()
    mean_cv = disp.groupby(["mode", "metric"])["dispersion"].mean()

    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    colors = mpl.colormaps["tab10"]
    plotted_any = False
    for idx, metric in enumerate(metrics):
        rows = []
        for mode in modes:
            frac = mode_values.get(mode, np.nan)
            y = mean_cv.get((mode, metric), np.nan)
            z = mean_perf.get((mode, metric), np.nan)
            if not (np.isfinite(frac) and np.isfinite and np.isfinite):
                continue
            rows.append((frac, y, z))
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[warn] plot_frac_cv_meanline: no finite data to plot.")
        return

    if all(str(m).startswith("sign_test") for m in modes):
        x_label = "sign_test frac"
    elif all(str(m).startswith("weight_test") for m in modes):
        x_label = "weight_test value"
    else:
        x_label = "mode value"

    # Use the full set of mode positions for x-limits.
    x_positions = [mode_values[m] for m in modes if np.isfinite(mode_values[m])]
    x_min, x_max = min(x_positions), max(x_positions)
    span = x_max - x_min
    pad = 0.05 * span if span > 0 else 1.0

    ax.set_xlabel(x_label, fontsize=18, labelpad=2)
    ax.set_ylabel("mean CV", fontsize=18, labelpad=2)
    ax.set_zlabel("mean performance", fontsize=18, labelpad=2)
    ax.set_xlim(x_min - pad, x_max + pad)
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "meanpoint_frac_cv_lines.png")
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(22, -35),
        back_view=(22, 145),
        dpi=600,
        tick_labelsize=16,
        tick_pad=-4,
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_weight_gauss_mean_cv(disp: pd.DataFrame, combined: pd.DataFrame, out_dir: str, show: bool = True):
    """
    3D line plot for weight_test (and other numeric modes):
      x = Gaussian magnitude (alpha) on log scale (log10 transform)
      y = mean CV (dispersion) per metric (MC/IPC/KR/GR)
      z = mean of an invariance column (prefers `cosine_similarity`)
    CV is not computed for the invariance column; we reuse the dispersion table for MC/IPC/KR/GR only.
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        mode_str = str(mode)
        nums = re.findall(r"[0-9]+(?:\\.[0-9]+)?", mode_str)
        for tok in nums:
            try:
                return float(tok)
            except Exception:
                continue
        return np.nan

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    if not mode_vals:
        print("[warn] plot_weight_gauss_mean_cv: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])
    modes = [m for m, _ in mode_vals]

    mean_tbl = _compute_mean_table(combined)
    inv_metric = next(
        (m for m in ("cosine_similarity", "wt_mean", "mean") if m in combined.columns),
        None,
    )
    if inv_metric is None:
        print("[warn] plot_weight_gauss_mean_cv: no invariance column found (need cosine_similarity/wt_mean/mean).")
        return
    inv_tbl = _compute_mean_table(combined, metrics=[inv_metric])
    mean_mean_lookup = (
        inv_tbl[inv_tbl["metric"] == inv_metric]
        .groupby("mode")["mean"]
        .mean()
    )
    if mean_mean_lookup.empty:
        print(f"[warn] plot_weight_gauss_mean_cv: no values for invariance metric '{inv_metric}'.")
        return
    z_label_base = "Cosine similarity" if inv_metric == "cosine_similarity" else "Wt mean"
    cv_lookup = disp.groupby(["mode", "metric"])["dispersion"].mean()

    # Metrics that have both mean and cv
    metrics = [m for m in ("MC", "IPC", "KR", "GR") if (m in mean_tbl["metric"].unique()) and (m in disp["metric"].unique())]
    if not metrics:
        print("[warn] plot_weight_gauss_mean_cv: no overlapping metrics with mean+CV.")
        return

    # Choose a scaling for the mean column so z-values stay in a readable range.
    max_mean = np.nanmax(mean_mean_lookup.values) if len(mean_mean_lookup) else np.nan
    if np.isfinite(max_mean) and max_mean > 0:
        z_power = int(np.floor(np.log10(max_mean)))
        z_scale = 10.0 ** z_power
    else:
        z_power = 0
        z_scale = 1.0

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    colors = mpl.colormaps["tab10"]
    plotted = False
    inv_vals_scaled = []

    for idx, metric in enumerate(metrics):
        rows = []
        for mode, x_raw in mode_vals:
            y_cv = cv_lookup.get((mode, metric), np.nan)
            z_mean = mean_mean_lookup.get(mode, np.nan)
            if not (np.isfinite(y_cv) and np.isfinite(z_mean) and x_raw > 0):
                continue
            inv_scaled = z_mean / z_scale
            rows.append((np.log10(x_raw), y_cv, inv_scaled))
            inv_vals_scaled.append(inv_scaled)
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        ax.plot(xs, zs, ys, marker="o", color=colors(idx % 10), label=metric)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_mean_cv: no finite data to plot.")
        return

    ax.set_xlabel("log10(Noise Magnitude)", fontsize=18, labelpad=10)
    ax.set_zlabel("mean CV ", fontsize=18, labelpad=12)
    if inv_metric == "cosine_similarity":
        ax.set_ylabel(z_label_base, fontsize=18, labelpad=10)
    else:
        wt_base = _format_wt_mean_axis_as_offset(ax, inv_vals_scaled)
        if wt_base is None:
            ax.set_ylabel(f"{z_label_base} (×1e{z_power})", fontsize=18, labelpad=10)
        else:
            wt_base_micro = wt_base * 1000.0
            ax.set_ylabel(
                f"ΔWt from {wt_base_micro:.0f} (×1e-6)",
                fontsize=18,
                labelpad=10,
            )
    # helpful x ticks at common magnitudes if they are within range
    xticks = [v for v in (1, 10, 100, 1000) if np.isfinite(np.log10(v))]
    ax.set_xticks(np.log10(xticks))
    ax.set_xticklabels([str(v) for v in xticks])
    #ax.legend()
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "weight_mean_cv_log3d.png")
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(22, -35),
        back_view=(22, 145),
        dpi=600,
        tick_labelsize=16,
        tick_pad=-2,
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_weight_gauss_mean_perf(disp: pd.DataFrame, combined: pd.DataFrame, out_dir: str, show: bool = True):
    """
    3D line plot like plot_weight_gauss_mean_cv but using mean performance instead of mean CV:
      x = Gaussian magnitude (alpha) on log scale
      y = mean performance (per metric MC/IPC/KR/GR)
      z = invariance column (prefers `cosine_similarity`)
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        mode_str = str(mode)
        nums = re.findall(r"[0-9]+(?:\\.[0-9]+)?", mode_str)
        for tok in nums:
            try:
                return float(tok)
            except Exception:
                continue
        return np.nan

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    if not mode_vals:
        print("[warn] plot_weight_gauss_mean_perf: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])

    mean_tbl = _compute_mean_table(combined)
    mean_lookup = mean_tbl.groupby(["mode", "metric"])["mean"].mean()
    inv_metric = next(
        (m for m in ("cosine_similarity", "wt_mean", "mean") if m in combined.columns),
        None,
    )
    if inv_metric is None:
        print("[warn] plot_weight_gauss_mean_perf: no invariance column found (need cosine_similarity/wt_mean/mean).")
        return
    inv_tbl = _compute_mean_table(combined, metrics=[inv_metric])
    mean_mean_lookup = (
        inv_tbl[inv_tbl["metric"] == inv_metric]
        .groupby("mode")["mean"]
        .mean()
    )
    if mean_mean_lookup.empty:
        print(f"[warn] plot_weight_gauss_mean_perf: no values for invariance metric '{inv_metric}'.")
        return
    z_label_base = "Cosine similarity" if inv_metric == "cosine_similarity" else "Wt mean"

    # Determine z scaling for nicer scientific-label axis
    max_mean = np.nanmax(mean_mean_lookup.values) if len(mean_mean_lookup) else np.nan
    if np.isfinite(max_mean) and max_mean > 0:
        z_power = int(np.floor(np.log10(max_mean)))
        z_scale = 10.0 ** z_power
    else:
        z_power = 0
        z_scale = 1.0

    # Metrics present in mean table (performance means)
    metrics = [m for m in ("MC", "IPC", "KR", "GR") if m in mean_tbl["metric"].unique()]
    if not metrics:
        print("[warn] plot_weight_gauss_mean_perf: no metrics with mean values found.")
        return

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    colors = mpl.colormaps["tab10"]
    plotted = False
    inv_vals_scaled = []

    for idx, metric in enumerate(metrics):
        rows = []
        for mode, x_raw in mode_vals:
            y_mean = mean_lookup.get((mode, metric), np.nan)
            z_mean = mean_mean_lookup.get(mode, np.nan)
            if not (np.isfinite(y_mean) and np.isfinite(z_mean) and x_raw > 0):
                continue
            inv_scaled = z_mean / z_scale
            rows.append((np.log10(x_raw), y_mean, inv_scaled))
            inv_vals_scaled.append(inv_scaled)
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        ax.plot(xs, zs, ys, marker="o", color=colors(idx % 10), label=metric)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_mean_perf: no finite data to plot.")
        return

    ax.set_xlabel("log10(Noise Magnitude)", fontsize=18, labelpad=10)
    ax.set_zlabel("Mean Metric Value", fontsize=18, labelpad=10)
    if inv_metric == "cosine_similarity":
        ax.set_ylabel(z_label_base, fontsize=18, labelpad=18)
    else:
        wt_base = _format_wt_mean_axis_as_offset(ax, inv_vals_scaled)
        if wt_base is None:
            ax.set_ylabel(f"{z_label_base} (×1e{z_power})", fontsize=18, labelpad=18)
        else:
            wt_base_micro = wt_base * 1000.0
            ax.set_ylabel(
                f"ΔWt from {wt_base_micro:.0f} (×1e-6)",
                fontsize=18,
                labelpad=18,
            )
    xticks = [v for v in (1, 10, 100, 1000) if np.isfinite(np.log10(v))]
    ax.set_xticks(np.log10(xticks))
    ax.set_xticklabels([str(v) for v in xticks])
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "weight_mean_perf_log3d.png")
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(22, -35),
        back_view=(22, 145),
        dpi=600,
        tick_labelsize=16,
        tick_pad=-2,
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_rho_cv_other_perf(
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
    model: str = "",
):
    """
    3D line plot at fixed spectral radius:
      x = spectral radius (rho_target)
      y = CV across non-rho hyperparameters (leak, input_scale) within each fixed rho
      z = mean performance at that fixed rho

    CV is computed per (mode, src, group_id, rho_target, metric) across leak/input_scale,
    then averaged across groups for each (rho_target, metric).
    """
    os.makedirs(out_dir, exist_ok=True)

    metric_cols = [m for m in ("MC", "IPC", "KR", "GR") if m in combined.columns]
    required_cols = ["mode", "src", "shuffle_id", "rho_target", "leak", "input_scale"] + metric_cols
    if not metric_cols:
        print("[warn] plot_rho_cv_other_perf: no MC/IPC/KR/GR columns found.")
        return
    if any(c not in combined.columns for c in required_cols):
        print("[warn] plot_rho_cv_other_perf: missing required columns; skipping.")
        return

    df = combined.copy()
    model = str(model or "").strip()
    if model:
        df = df[df["mode"].astype(str) == model].copy()
        if df.empty:
            print()
            modes = sorted(combined["mode"].astype(str).dropna().unique().tolist())
            print(modes)
            preview = ", ".join(modes[:12]) + (" ..." if len(modes) > 12 else "")
            print(f"[warn] plot_rho_cv_other_perf: no rows for model='{model}'.")
            print(f"[info] available models: {preview}")
            return

    for c in ("rho_target", "leak", "input_scale", *metric_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["rho_target"])].copy()
    if df.empty:
        print("[warn] plot_rho_cv_other_perf: no finite rho_target values.")
        return

    df = _assign_group_ids(df)
    df_agg = _aggregate_over_hparams(df, metric_cols)
    if df_agg.empty:
        print("[warn] plot_rho_cv_other_perf: no aggregated rows after deduplication.")
        return

    keys = ["mode", "src", "group_id", "rho_target", "leak", "input_scale"]
    df_long = df_agg.melt(id_vars=keys, value_vars=metric_cols, var_name="metric", value_name="value")

    per_group_rho = (
        df_long.groupby(["mode", "src", "group_id", "rho_target", "metric"], as_index=False)
        .agg(
            cv_other=("value", lambda x: _dispersion(x.to_numpy())),
            perf_mean=("value", "mean"),
            n_other_hparams=("value", "size"),
        )
    )

    summary = (
        per_group_rho.groupby(["rho_target", "metric"], as_index=False)
        .agg(
            cv_other=("cv_other", "mean"),
            perf_mean=("perf_mean", "mean"),
            n_groups=("group_id", "size"),
        )
    )
    if summary.empty:
        print("[warn] plot_rho_cv_other_perf: no data to plot.")
        return

    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    colors = mpl.colormaps["tab10"]
    plotted = False

    for idx, metric in enumerate(metric_cols):
        sub = summary[summary["metric"] == metric].copy()
        if sub.empty:
            continue
        sub = sub[np.isfinite(sub["rho_target"]) & np.isfinite(sub["cv_other"]) & np.isfinite(sub["perf_mean"])]
        if sub.empty:
            continue
        sub = sub.sort_values("rho_target")
        xs = sub["rho_target"].to_numpy(float)
        ys = sub["cv_other"].to_numpy(float)
        zs = sub["perf_mean"].to_numpy(float)
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] plot_rho_cv_other_perf: no finite data to plot.")
        return

    rho_vals = sorted(summary["rho_target"].dropna().unique())
    if rho_vals:
        ax.set_xticks(rho_vals)
    ax.set_xlabel("spectral radius (rho_target)", fontsize=18, labelpad=8)
    ax.set_ylabel("CV of non-rho hyperparameters", fontsize=14, labelpad=8)
    ax.set_zlabel("mean performance", fontsize=18, labelpad=8)
    if model:
        ax.set_title(f"Performance vs non-rho CV at fixed spectral radius ({model})", fontsize=14)
    else:
        ax.set_title("Performance vs non-rho CV at fixed spectral radius", fontsize=16)
    ax.legend()
    _tight_layout_quiet(fig)

    if model:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
        out_name = f"rho_cv_other_perf_3d.{safe_model}.png"
    else:
        out_name = "rho_cv_other_perf_3d.png"
    out_fig = os.path.join(out_dir, out_name)
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(22, -35),
        back_view=(22, 145),
        dpi=600,
        tick_labelsize=16,
        tick_pad=-2,
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")

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
        #"cel_sample", temporaroly removed
        "conn_shuf_only",
        "local_sign+flat",
        "local_sign+sample",
        "local_sign+binary",
        "global_sign_pres",
        "binary+shuffle",

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
        #"cel_sample": "#ff6565",
        "conn_shuf_only": "#ff9752",
        "local_sign+flat":  "#002CF1",
        "local_sign+sample": "#AF7DF5",
        "local_sign+binary": "#8ADEF3",
        "global_sign_pres" : "#1C373D",
        "binary+shuffle": "#FF00FF",

    }
    if not metrics:
        return
    plt.rcParams.update({
        "axes.titlesize": 25,
        "axes.labelsize": 25,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
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
                    "binary+shuffle": "Binary sign + shuffle",



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
            ax.set_title(f"Invariance of {m}")
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

        if False:
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

        #fig.tight_layout(rect=(0.00, 0.00, 0.96, 0.85))
        fig.tight_layout(rect=(0.01, 0.00, 1.0, 1.0))
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
    
                    # pg.tost can emit overflow RuntimeWarnings for extreme t values; suppress locally
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning, module="pingouin")
                        tost = (pg.tost(vals_a, vals_b, np.abs(np.median(vals_all)) * 0.05, paired=False))
                    if tost['pval'].iloc[0]<0.05:
                        print(rf"{m} & {a} vs {b} & {tost['bound'].iloc[0]:.4g} & {tost['pval'].iloc[0]:.4g} \\")

        #print(f"\n{m}")
        #print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))  # one table per metric
        #print(f"H = {H:.4g}, p = {p:.4g}")
    print()


# --------------------------- main ----------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    wants_compare = bool(args.compare_mean_a or args.compare_mean_b or args.compare_only)
    if wants_compare:
        if not (args.compare_mean_a and args.compare_mean_b):
            raise ValueError("Both --compare-mean-a and --compare-mean-b are required for comparison mode.")
        if not os.path.isfile(args.compare_mean_a):
            raise FileNotFoundError(f"Comparison mean table not found: {args.compare_mean_a}")
        if not os.path.isfile(args.compare_mean_b):
            raise FileNotFoundError(f"Comparison mean table not found: {args.compare_mean_b}")

        out_cmp = args.compare_mean_out or _safe_path(
            os.path.join(args.out_dir, "mode_metric_mean_comparison.csv")
        )
        comp_tbl = compare_mode_metric_means(
            args.compare_mean_a,
            args.compare_mean_b,
            out_cmp,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
            value_col=args.compare_value_col,
        )
        out_cmp_plot = args.compare_plot_out or _safe_path(
            os.path.join(args.out_dir, "mode_metric_mean_self_drop.png")
        )
        metric_filter = _normalize_compare_metrics(args.compare_metric)
        plot_tbl = comp_tbl
        if metric_filter:
            plot_tbl = comp_tbl[comp_tbl["metric"].isin(metric_filter)].copy()
            if plot_tbl.empty:
                available = sorted(comp_tbl["metric"].dropna().unique())
                raise ValueError(
                    f"--compare-metric={args.compare_metric!r} did not match any metrics. "
                    f"Available: {available}"
                )
        plot_mode_self_drop_comparison(
            plot_tbl,
            out_cmp_plot,
            label_a=args.compare_label_a,
            label_b=args.compare_label_b,
        )
        if args.compare_tost_preservation:
            ref_tbl, ref_col = _read_compare_table(args.compare_mean_a, value_col=args.compare_value_col)
            new_tbl, new_col = _read_compare_table(args.compare_mean_b, value_col=args.compare_value_col)
            if ref_col != new_col:
                raise ValueError(
                    f"TOST preservation needs same value column on both inputs, got: {ref_col} vs {new_col}"
                )
            out_tost = args.compare_tost_out or _safe_path(
                os.path.join(args.out_dir, "tost_preservation_summary.csv")
            )
            print_tost_preservation_summary(
                ref_tbl,
                new_tbl,
                value_col=ref_col,
                label_ref=args.compare_label_a,
                label_new=args.compare_label_b,
                alpha=float(args.compare_tost_alpha),
                bound_frac=float(args.compare_tost_bound_frac),
                metrics_filter=metric_filter,
                out_csv=out_tost,
            )
        if args.compare_only:
            return

    if not os.path.isfile(args.combined):
        raise FileNotFoundError(f"Combined CSV not found: {args.combined}")

    combined = _read_combined_csv(args.combined)
    combined = _ensure_columns(combined)

    # Save a copy (non-destructive; versioned if exists)
    out_comb = _safe_path(os.path.join(args.out_dir, "combined.ALL.csv"))
    #combined.to_csv(out_comb, index=False)
    #print(f"[saved] {out_comb}  (rows={len(combined)})")

    # Compute and save dispersion table
    disp = _compute_dispersion_table(combined,mode="cv")
    #out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    #disp.to_csv(out_disp, index=False)

    # Plots
    #plot_frac_arch_histograms(disp, args.out_dir, args.bins)
    #plot_frac_cv_meanline(disp, combined, args.out_dir)
    #plot_weight_gauss_mean_cv(disp, combined, args.out_dir, show=True)
    #plot_weight_gauss_mean_perf(disp, combined, args.out_dir, show=True)
    #plot_rho_cv_other_perf(combined, args.out_dir, show=True, model=args.model)
    plot_overlaid_arch_histograms(disp, args.out_dir, args.bins)
    #plot_mc_vs_gr_all_arch(combined, args.out_dir, args.scatter_alpha)
    #print_kruskal_wallis_tables(disp)


if __name__ == "__main__":
    main()
