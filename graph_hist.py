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
from matplotlib.ticker import FormatStrFormatter

if not os.environ.get("MPLBACKEND"):
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from util.graph_utils import (
    _compute_dispersion_table,
    _compute_mean_table,
    _assign_group_ids,
    _aggregate_over_hparams,
)

# Optional hard filter for analysis modes.
# Leave empty to analyze all modes, or populate with exact mode names.
ANALYSIS_MODE_FILTER = [
    #"binary_base_topology_shuffle",
    #"binary_base"
    #------------------------------
    #"binary+shuffle",
    #"local_sign+binary",
    #"global_sign_pres",
    #"binary+conshuffle+wshuffle"
    #------------------------------
    "real",
    "shuffle",
    "celW+connShuf",
    "conn_shuf_only",
    #-------- main experiment above was shuffle experiment
      #          "real",
      #      "cel+randN",
      #      "er+randN",
      #      "ws_p0.1+randN",
      #      "local_sign", ##this is n(0,1) model
      #      "local_sign+flat",
      #      "local_sign+binary",
      #      "global_sign_pres",
      #      "binary_base",
]

SHORT_THESIS_NAMES = {
    "real": "C. elegans",
    "shuffle": "Weight shuffle",
    "shuffle_weights": "Weight shuffle",
    "celW+connShuf": "Conn. + wt. shuffle",
    "conn_shuf": "Conn. + wt. shuffle",
    "conn_shuf_only": "Connection shuffle",
    "local_sign+binary": "Sign-pres. pm1 wt.",
    "global_sign_pres": "Sign-pres. pm1 + wt. shuf.",
    "global_sign_pres_real_w": "Sign-pres. real wt.",
    "global_sign_pres_real_weight": "Sign-pres. real wt.",
    "binary+conshuffle+wshuffle": "pm1 sign-pres. + shuf.",
    "binary_base": "Binary wt.",
    "binary_base_topology_shuffle": "Binary wt. + shuf.",
    "binary+shuffle": "pm1 + conn. shuffle",
    "cel_randN": "CE Gaussian wt.",
    "er_randN": "ER Gaussian wt.",
    "ws_p0.1+randN": "WS Gaussian wt.",
    "ws_p01_randN": "WS Gaussian wt.",
    "local_sign": "Sign-pres. Gaussian",
    "local_sign+flat": "Sign-pres. uniform",
}

_SHORT_THESIS_NAME_ALIASES = {
    "shuffle": "shuffle_weights",
    "cel+randN": "cel_randN",
    "er+randN": "er_randN",
    "ws_p0.1+randN": "ws_p01_randN",
    "celW+connShuf": "conn_shuf",
}


def _short_thesis_name(mode):
    if not isinstance(mode, str):
        return mode
    canonical = _SHORT_THESIS_NAME_ALIASES.get(mode, mode)
    return SHORT_THESIS_NAMES.get(canonical, SHORT_THESIS_NAMES.get(mode, mode))


def _short_legend_name(mode):
    if not isinstance(mode, str):
        return mode
    legend_only = {
        "cel_sample": "Sampled wt.",
        "local_sign+sample": "Sign-pres. sampled",
    }
    canonical = _SHORT_THESIS_NAME_ALIASES.get(mode, mode)
    return legend_only.get(canonical, legend_only.get(mode, _short_thesis_name(mode)))

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
    ap.add_argument("--frac-cv-bins", type=int, default=4,
                    help="Mean-CV axis tick bins for the frac/CV/performance mean-line plot.")
    ap.add_argument("--scatter-alpha", type=float, default=0.55,
                    help="Alpha for MC-vs-GR scatter.")
    ap.add_argument(
        "--model",
        default="",
        help="Optional exact mode/model name filter (e.g., 'weight_test10.0') for rho/CV/performance plot.",
    )
    ap.add_argument(
        "--rho-cv-drop-kr-gr",
        action="store_true",
        help="For rho/CV/performance plot, exclude KR and GR curves (plot MC/IPC only).",
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
        help="Optional metric filter for comparison plot (e.g., 'cosine_similarity', 'covariance', or 'kl_to_gaussian').",
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
    ap.add_argument(
        "--local-sign-binary-csv",
        default="",
        help="Optional extra combined CSV used only to overlay mode='real' as a wall on weight-gauss plots.",
    )
    ap.add_argument(
        "--weight-gauss-cv-only",
        action="store_true",
        help="Plot weight-test modes as 2D interpolation fraction vs mean CV and mean metric value; do not use KL.",
    )
    ap.add_argument(
        "--sign-norm-ablation",
        action="store_true",
        help="Plot the sign-balance normalization ablation and exit before default histogram plots.",
    )
    ap.add_argument(
        "--sign-norm-prefix",
        default="sign_test_og_cel",
        help="Mode prefix used to identify sign-normalization ablation rows.",
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


def _replace_path(path: str) -> str:
    """Use a stable output path, replacing an older plot/table if it exists."""
    return path


def _mode_numeric_value(mode: str) -> float:
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(mode))
    for tok in nums:
        try:
            return float(tok)
        except Exception:
            continue
    return np.nan


def _write_weight_metric_difference_table(
    y_lookup,
    y_label: str,
    out_dir: str,
    tex_name: str,
    mode_vals,
    metrics,
    zero_mode=None,
    overlay_lookup=None,
    log_prefix: str = "weight_metric_difference_table",
    mode_value_symbol: str = r"\alpha",
):
    baseline_info = {}
    for metric in metrics:
        baseline_value = np.nan
        baseline_mode = ""
        baseline_source = ""

        if overlay_lookup is not None:
            baseline_value = overlay_lookup.get(("real", metric), np.nan)
            if np.isfinite(baseline_value):
                baseline_mode = "real"
                baseline_source = "overlay"

        if zero_mode is not None:
            zero_value = y_lookup.get((zero_mode, metric), np.nan)
            if np.isfinite(zero_value):
                baseline_value = zero_value
                baseline_mode = zero_mode
                baseline_source = "zero-noise mode"

        if not np.isfinite(baseline_value):
            for mode, _ in mode_vals:
                cand = y_lookup.get((mode, metric), np.nan)
                if np.isfinite(cand):
                    baseline_value = cand
                    baseline_mode = str(mode)
                    baseline_source = "first finite mode"
                    break

        if np.isfinite(baseline_value):
            baseline_info[metric] = {
                "baseline_value": float(baseline_value),
                "baseline_mode": baseline_mode,
                "baseline_source": baseline_source,
            }

    if not baseline_info:
        return

    def _latex_escape(text: str) -> str:
        out = str(text)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "_": r"\_",
            "#": r"\#",
            "$": r"\$",
            "{": r"\{",
            "}": r"\}",
        }
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        return out

    def _fmt_num(x: float, digits: int = 3) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{x:.{digits}g}"

    def _fmt_signed(x: float, digits: int = 3) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{x:+.{digits}g}"

    def _fmt_mode_value(raw_value: float) -> str:
        if not np.isfinite(raw_value):
            return "NA"
        if abs(raw_value - round(raw_value)) < 1e-12:
            return str(int(round(raw_value)))
        return f"{raw_value:.3g}"

    data_modes = []
    for mode, raw_value in mode_vals:
        if mode == zero_mode:
            continue
        if any(np.isfinite(y_lookup.get((mode, metric), np.nan)) for metric in metrics):
            data_modes.append((mode, raw_value))

    lines = [
        "\\begin{tabular}{" + ("l" + "c" * (1 + len(data_modes))) + "}",
        "\\toprule",
    ]

    header_cells = ["Metric", "Baseline"]
    for _, raw_value in data_modes:
        header_cells.append(f"${mode_value_symbol}={_fmt_mode_value(raw_value)}$")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append("\\midrule")

    for metric in metrics:
        info = baseline_info.get(metric)
        if info is None:
            continue
        baseline_value = info["baseline_value"]
        denom = abs(baseline_value)
        row_cells = [_latex_escape(metric), _fmt_num(baseline_value, digits=4)]
        for mode, _ in data_modes:
            value = y_lookup.get((mode, metric), np.nan)
            if not np.isfinite(value):
                row_cells.append("--")
                continue
            delta = float(value - baseline_value)
            if denom > 1e-12:
                pct_delta = 100.0 * delta / denom
                cell = f"{_fmt_signed(delta)} ({_fmt_signed(pct_delta)}\\%)"
            else:
                cell = _fmt_signed(delta)
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    latex_table = "\n".join(lines)

    out_tex = _safe_path(os.path.join(out_dir, tex_name))
    with open(out_tex, "w", encoding="utf-8") as handle:
        handle.write(latex_table + "\n")

    print(f"[latex] {log_prefix}: {y_label}")
    print(latex_table)
    print(f"[saved] {out_tex}")


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
            "neuron_bias",
            "MC",
            "IPC",
            "KR",
            "GR",
            "raw_rho",
            "ref_rho",
            "post_rho",
            "scale_factor",
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
        "kl": "kl_to_gaussian",
        "kl_gaussian": "kl_to_gaussian",
    }
    out = []
    for tok in str(spec).split(","):
        key = tok.strip()
        if not key:
            continue
        out.append(aliases.get(key.lower(), key))
    return out


def _load_local_sign_binary_overlay(local_csv: str):
    if not local_csv:
        return None
    if not os.path.isfile(local_csv):
        print(f"[warn] real-wall overlay CSV not found: {local_csv}")
        return None

    extra = _read_combined_csv(local_csv)
    extra = _ensure_columns(extra)
    extra = extra[extra["mode"].astype(str) == "real"].copy()
    if extra.empty:
        print(f"[warn] real-wall overlay: no mode='real' rows in {local_csv}")
        return None

    disp_extra = _compute_dispersion_table(extra, mode="cv")
    mean_extra = _compute_mean_table(extra)
    out = {
        "cv_lookup": disp_extra.groupby(["mode", "metric"])["dispersion"].mean(),
        "mean_lookup": mean_extra.groupby(["mode", "metric"])["mean"].mean(),
        "kl_value": np.nan,
    }
    if "kl_to_gaussian" in extra.columns:
        kl_tbl = _compute_mean_table(extra, metrics=["kl_to_gaussian"])
        kl_lookup = kl_tbl.groupby("mode")["mean"].mean()
        out["kl_value"] = float(kl_lookup.get("real", np.nan))
    else:
        print("[warn] real-wall overlay missing 'kl_to_gaussian'; using KL baseline z=0.0 for overlay wall.")
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

    metric_order = ["MC", "IPC", "KR", "GR", "kl_to_gaussian", "cosine_similarity", "wt_mean"]
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
        ref_mean_lookup = (
            ref.groupby(["metric", "mode"], dropna=False)[value_col]
            .mean()
            .to_dict()
        )
        new_mean_lookup = (
            new.groupby(["metric", "mode"], dropna=False)[value_col]
            .mean()
            .to_dict()
        )

        def _fmt_mean(v):
            return f"{v:.6g}" if np.isfinite(v) else "NA"

        not_preserved = not_preserved.sort_values(["metric", "mode_a", "mode_b"], kind="stable")
        print("[tost] not preserved comparisons (baseline significant, new not significant):")
        for _, row in not_preserved.iterrows():
            p_ref = row.get(p_ref_col, np.nan)
            p_new = row.get(p_new_col, np.nan)
            mode_a = row["mode_a"]
            mode_b = row["mode_b"]
            mode_a_label = _short_thesis_name(mode_a)
            mode_b_label = _short_thesis_name(mode_b)
            metric = row["metric"]
            ref_mean_a = ref_mean_lookup.get((metric, mode_a), np.nan)
            ref_mean_b = ref_mean_lookup.get((metric, mode_b), np.nan)
            new_mean_a = new_mean_lookup.get((metric, mode_a), np.nan)
            new_mean_b = new_mean_lookup.get((metric, mode_b), np.nan)
            if np.isfinite(p_new):
                print(
                    f"[tost] {metric}: {mode_a_label} vs {mode_b_label} "
                    f"(p_{label_ref}={p_ref:.4g}, p_{label_new}={p_new:.4g}, bound={row['bound_ref']:.4g}, "
                    f"means_{label_ref}=[{mode_a_label}:{_fmt_mean(ref_mean_a)}, {mode_b_label}:{_fmt_mean(ref_mean_b)}], "
                    f"means_{label_new}=[{mode_a_label}:{_fmt_mean(new_mean_a)}, {mode_b_label}:{_fmt_mean(new_mean_b)}])"
                )
            else:
                print(
                    f"[tost] {metric}: {mode_a_label} vs {mode_b_label} "
                    f"(p_{label_ref}={p_ref:.4g}, p_{label_new}=NA, bound={row['bound_ref']:.4g}, "
                    f"means_{label_ref}=[{mode_a_label}:{_fmt_mean(ref_mean_a)}, {mode_b_label}:{_fmt_mean(ref_mean_b)}], "
                    f"means_{label_new}=[{mode_a_label}:{_fmt_mean(new_mean_a)}, {mode_b_label}:{_fmt_mean(new_mean_b)}])"
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


def _style_3d_axis(
    ax,
    tick_labelsize: int = 11,
    tick_pad: int = 2,
    pane_fill: bool = True,
    pane_facecolor=(1.0, 1.0, 1.0, 1.0),
    pane_edgecolor=(0.85, 0.85, 0.85, 1.0),
):
    """Apply a cleaner, publication-friendly 3D style."""
    ax.grid(True, which="major", linestyle=":", alpha=0.28)
    ax.tick_params(axis="x", which="major", labelsize=tick_labelsize, pad=-2)
    ax.tick_params(axis="y", which="major", labelsize=tick_labelsize, pad=-2)
    ax.tick_params(axis="z", which="major", labelsize=tick_labelsize, pad=0)
    y_formatter = ax.yaxis.get_major_formatter()
    if isinstance(y_formatter, mpl.ticker.ScalarFormatter):
        y_formatter.set_useOffset(False)
        y_formatter.set_scientific(False)
    ax.yaxis.get_offset_text().set_visible(False)

    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = pane_fill
            axis.pane.set_facecolor(pane_facecolor)
            axis.pane.set_edgecolor(pane_edgecolor)
    except Exception:
        pass


def _clear_3d_axis_background(ax):
    """Make the 3D axis background transparent while keeping guide lines visible."""
    try:
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.patch.set_alpha(0.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
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


def _pad_width_center(img: np.ndarray, target_w: int) -> np.ndarray:
    """Pad image to target width with white background, centered horizontally."""
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    out = np.full((h, target_w, 4), 255, dtype=np.uint8)
    left = (target_w - w) // 2
    out[:, left:left + w] = img
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
    pane_fill: bool = True,
    pane_facecolor=(1.0, 1.0, 1.0, 1.0),
    pane_edgecolor=(0.85, 0.85, 0.85, 1.0),
):
    """Save one publication PNG with front/back views side-by-side."""
    if isinstance(axes, (list, tuple, np.ndarray)):
        axes_list = [ax for ax in axes if ax is not None]
    else:
        axes_list = [axes] if axes is not None else []

    for ax in axes_list:
        _style_3d_axis(
            ax,
            tick_labelsize=tick_labelsize,
            tick_pad=tick_pad,
            pane_fill=pane_fill,
            pane_facecolor=pane_facecolor,
            pane_edgecolor=pane_edgecolor,
        )

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
    needed = ["mode","shuffle_id","rho_target","leak","input_scale","neuron_bias","MC","IPC","KR","GR","src"]
    df = df.copy()
    for c in needed:
        if c not in df.columns:
            if c in ("MC","IPC","KR","GR"):
                df[c] = np.nan
            elif c == "neuron_bias":
                df[c] = 0.0
            elif c == "shuffle_id":
                df[c] = -1
            elif c == "mode":
                df[c] = "unknown"
            elif c == "src":
                df[c] = "unknown"
            else:
                raise ValueError(f"Missing required column: {c}")
    # types
    df["mode"] = df["mode"].astype(str).str.strip()
    extras = [c for c in df.columns if c not in needed]
    return df[needed + extras].copy()


def _filter_to_modes(df: pd.DataFrame, modes=None, label: str = "data") -> pd.DataFrame:
    """Restrict dataframe to an explicit mode allowlist."""
    if modes is None:
        modes = ANALYSIS_MODE_FILTER
    allow = [str(m).strip() for m in (modes or []) if str(m).strip()]
    if not allow:
        return df
    mode_values = df["mode"].astype(str).str.strip()
    mode_counts = mode_values.value_counts(dropna=False)
    out = df[mode_values.isin(allow)].copy()
    print(f"[info] {label}: mode filter active ({len(allow)} mode(s)); kept {len(out)}/{len(df)} rows")
    requested_counts = {mode: int(mode_counts.get(mode, 0)) for mode in allow}
    print(
        f"[info] {label}: requested mode counts: "
        + ", ".join(f"{mode}={count}" for mode, count in requested_counts.items())
    )
    missing = [mode for mode, count in requested_counts.items() if count == 0]
    if missing:
        available = sorted(mode_values.dropna().unique().tolist())
        preview = ", ".join(available[:16]) + (" ..." if len(available) > 16 else "")
        print(f"[warn] {label}: requested mode(s) absent from input: {', '.join(missing)}")
        print(f"[info] {label}: available modes: {preview}")
    return out


def _dispersion(a: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    m = float(np.mean(a))
    s = float(np.std(a))
    return s/(abs(m)+1e-12) ## allows us to calcualte variance accross different models which have different scales

def _unique_hparam_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["rho_target","leak","input_scale","neuron_bias"]
    metrics = [c for c in ("MC","IPC","KR","GR") if c in df.columns]
    if not metrics:
        return df.copy()
    return (df.groupby(keys, as_index=False)[metrics]
              .mean()
              .sort_values(keys)
              .reset_index(drop=True))


def _sign_norm_label(norm_name: str) -> str:
    labels = {
        "spectral_radius": r"$W / \rho(W)$",
        "original_radius": r"$W / \rho(W_{\mathrm{orig}})$",
    }
    return labels.get(str(norm_name), str(norm_name))


def _parse_sign_fraction_from_mode(mode: str, prefix: str) -> float:
    mode_base = str(mode).split("__norm_", 1)[0]
    prefix = str(prefix or "").strip()
    if prefix:
        m = re.search(
            rf"{re.escape(prefix)}([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            mode_base,
        )
        if m:
            return float(m.group(1))
        return np.nan
    return _mode_numeric_value(mode_base)


def _infer_normalization_from_mode(mode: str) -> str:
    m = re.search(r"__norm_([A-Za-z0-9_+-]+)$", str(mode))
    if m:
        return m.group(1)
    return "spectral_radius"


def _prepare_sign_norm_ablation_df(combined: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = combined.copy()
    if "normalization" not in df.columns:
        df["normalization"] = ""
    df["normalization"] = df["normalization"].astype(str).str.strip()
    inferred_norm = df["mode"].astype(str).map(_infer_normalization_from_mode)
    df.loc[df["normalization"].isin(["", "nan", "None"]), "normalization"] = inferred_norm

    mode_base = df["mode"].astype(str).str.split("__norm_", n=1).str[0]
    if prefix:
        df = df[mode_base.str.startswith(str(prefix))].copy()
    if df.empty:
        return df

    df["sign_frac"] = df["mode"].map(lambda m: _parse_sign_fraction_from_mode(m, prefix))
    df = df[np.isfinite(df["sign_frac"])].copy()
    for col in (
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
        "MC",
        "IPC",
        "KR",
        "GR",
        "raw_rho",
        "ref_rho",
        "post_rho",
        "scale_factor",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _summarize_group_values(
    per_group: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    if per_group.empty:
        return pd.DataFrame(columns=group_cols + ["mean", "std", "sem", "n_groups"])
    summary = (
        per_group.groupby(group_cols, as_index=False)[value_col]
        .agg(mean="mean", std="std", n_groups="count")
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n_groups"].clip(lower=1))
    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary


def _save_sign_norm_metric_grid(
    summary: pd.DataFrame,
    out_path: str,
    *,
    value_label: str,
    title: str,
    metric_order: list[str],
    norm_order: list[str],
):
    if summary.empty:
        print(f"[warn] {title}: no summary rows to plot.")
        return
    metrics = [m for m in metric_order if m in set(summary["metric"])]
    if not metrics:
        print(f"[warn] {title}: no requested metrics present.")
        return

    summary = summary[summary["normalization"].astype(str).isin(norm_order)].copy()
    if summary.empty:
        print(f"[warn] {title}: no rows for requested normalization modes.")
        return

    colors = {
        "spectral_radius": "#1f77b4",
        "original_radius": "#d95f02",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=300, squeeze=False)
    axes_flat = axes.ravel()
    handles = []
    labels = []

    for ax_idx, metric in enumerate(metrics[:4]):
        ax = axes_flat[ax_idx]
        plotted = False
        for norm_name in norm_order:
            sub = summary[
                (summary["metric"] == metric)
                & (summary["normalization"].astype(str) == norm_name)
            ].sort_values("sign_frac")
            if sub.empty:
                continue
            xs = sub["sign_frac"].to_numpy(float)
            ys = sub["mean"].to_numpy(float)
            sem = sub["sem"].to_numpy(float)
            line = ax.errorbar(
                xs,
                ys,
                yerr=sem,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                capsize=2.5,
                color=colors.get(norm_name),
                label=_sign_norm_label(norm_name),
            )
            if ax_idx == 0:
                handles.append(line)
                labels.append(_sign_norm_label(norm_name))
            plotted = True
        if not plotted:
            ax.set_axis_off()
            continue
        ax.set_title(metric)
        ax.set_xlabel("negative edge fraction")
        ax.set_ylabel(value_label)
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.grid(True, alpha=0.22)
        ax.margins(y=0.08)

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(title, y=0.985, fontsize=16)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(2, len(handles)),
            frameon=False,
        )
    fig.align_ylabels(axes_flat)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.89])
    out_path = _replace_path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _save_sign_norm_scaling_grid(
    summary: pd.DataFrame,
    out_path: str,
    *,
    norm_order: list[str],
):
    if summary.empty:
        print("[warn] sign normalization scaling: no summary rows to plot.")
        return
    diag_cols = [
        ("post_rho", "post spectral radius"),
        ("ref_rho", "reference spectral radius"),
        ("raw_rho", "raw spectral radius"),
        ("scale_factor", "scale factor"),
    ]
    summary = summary[summary["normalization"].astype(str).isin(norm_order)].copy()
    if summary.empty:
        print("[warn] sign normalization scaling: no rows for requested normalization modes.")
        return

    colors = {
        "spectral_radius": "#1f77b4",
        "original_radius": "#d95f02",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=300, squeeze=False)
    axes_flat = axes.ravel()
    handles = []
    labels = []

    for ax_idx, (col, label) in enumerate(diag_cols):
        ax = axes_flat[ax_idx]
        if col not in summary.columns:
            ax.set_axis_off()
            continue
        plotted = False
        for norm_name in norm_order:
            sub = summary[summary["normalization"].astype(str) == norm_name].sort_values("sign_frac")
            if sub.empty:
                continue
            xs = sub["sign_frac"].to_numpy(float)
            ys = sub[col].to_numpy(float)
            sem_col = f"{col}_sem"
            yerr = sub[sem_col].to_numpy(float) if sem_col in sub.columns else None
            line = ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                capsize=2.5,
                color=colors.get(norm_name),
                label=_sign_norm_label(norm_name),
            )
            if ax_idx == 0:
                handles.append(line)
                labels.append(_sign_norm_label(norm_name))
            plotted = True
        if not plotted:
            ax.set_axis_off()
            continue
        ax.set_title(label)
        ax.set_xlabel("negative edge fraction")
        ax.set_ylabel(label)
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.grid(True, alpha=0.22)
        ax.margins(y=0.08)

    fig.suptitle("Sign-balance normalization scaling diagnostics", y=0.985, fontsize=16)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(2, len(handles)),
            frameon=False,
        )
    fig.align_ylabels(axes_flat)
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.89])
    out_path = _replace_path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def _save_sign_norm_combined_grid(
    performance: pd.DataFrame,
    cv: pd.DataFrame,
    scaling: pd.DataFrame,
    out_path: str,
    *,
    metric_order: list[str],
    norm_order: list[str],
):
    """Save performance, CV, and radius diagnostics as one compact figure."""
    norm_colors = {
        "spectral_radius": "#1f77b4",
        "original_radius": "#d95f02",
    }
    norm_styles = {
        "spectral_radius": "-",
        "original_radius": "--",
    }
    norm_markers = {
        "spectral_radius": "o",
        "original_radius": "s",
    }
    metric_colors = {
        "MC": "#1f77b4",
        "IPC": "#d95f02",
        "KR": "#2ca02c",
        "GR": "#9467bd",
    }
    metrics = [metric for metric in metric_order if metric in set(performance["metric"])]
    if not metrics:
        print("[warn] combined sign normalization figure: no metrics to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.4), dpi=300, squeeze=False)
    memory_performance_ax, kernel_performance_ax, rho_ax = axes[0]
    memory_cv_ax, kernel_cv_ax, scale_ax = axes[1]

    def draw_combined_metrics(
        summary: pd.DataFrame,
        ax: plt.Axes,
        selected_metrics: list[str],
        ylabel: str,
    ):
        for metric in selected_metrics:
            for norm_name in norm_order:
                sub = summary[
                    (summary["metric"] == metric)
                    & (summary["normalization"].astype(str) == norm_name)
                ].sort_values("sign_frac")
                if sub.empty:
                    continue
                xs = sub["sign_frac"].to_numpy(float)
                ys = sub["mean"].to_numpy(float)
                sem = sub["sem"].to_numpy(float)
                ax.plot(
                    xs,
                    ys,
                    color=metric_colors[metric],
                    linestyle=norm_styles[norm_name],
                    marker=norm_markers[norm_name],
                    linewidth=1.8,
                    markersize=3.6,
                )
                ax.fill_between(
                    xs,
                    ys - sem,
                    ys + sem,
                    color=metric_colors[metric],
                    alpha=0.06,
                    linewidth=0,
                )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22)
        ax.margins(y=0.08)

    memory_metrics = [metric for metric in ("MC", "IPC") if metric in metrics]
    kernel_metrics = [metric for metric in ("KR", "GR") if metric in metrics]
    draw_combined_metrics(
        performance,
        memory_performance_ax,
        memory_metrics,
        "Mean performance",
    )
    memory_performance_ax.set_title("Memory metrics: mean performance")
    draw_combined_metrics(
        performance,
        kernel_performance_ax,
        kernel_metrics,
        "Mean performance",
    )
    kernel_performance_ax.set_title("Kernel metrics: mean performance")

    draw_combined_metrics(cv, memory_cv_ax, memory_metrics, "Mean within-target CV")
    memory_cv_ax.set_title("Memory metrics: hyperparameter CV")
    draw_combined_metrics(cv, kernel_cv_ax, kernel_metrics, "Mean within-target CV")
    kernel_cv_ax.set_title("Kernel metrics: hyperparameter CV")

    if scaling.empty:
        rho_ax.set_axis_off()
        scale_ax.set_axis_off()
    else:
        first_norm = norm_order[0]
        base = scaling[
            scaling["normalization"].astype(str) == first_norm
        ].sort_values("sign_frac")
        for value_col, label, color, linestyle in (
            ("raw_rho", r"$\rho(W)$", "#666666", "-."),
            ("ref_rho", r"$\rho(W_{\mathrm{orig}})$", "#111111", ":"),
        ):
            if value_col in base.columns and not base.empty:
                rho_ax.plot(
                    base["sign_frac"].to_numpy(float),
                    base[value_col].to_numpy(float),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=label,
                )
        for norm_name in norm_order:
            sub = scaling[
                scaling["normalization"].astype(str) == norm_name
            ].sort_values("sign_frac")
            if sub.empty:
                continue
            rho_ax.plot(
                sub["sign_frac"].to_numpy(float),
                sub["post_rho"].to_numpy(float),
                color=norm_colors[norm_name],
                linestyle=norm_styles[norm_name],
                marker=norm_markers[norm_name],
                linewidth=1.8,
                markersize=3.6,
                label=rf"post $\rho$: {_sign_norm_label(norm_name)}",
            )
            scale_ax.plot(
                sub["sign_frac"].to_numpy(float),
                sub["scale_factor"].to_numpy(float),
                color=norm_colors[norm_name],
                linestyle=norm_styles[norm_name],
                marker=norm_markers[norm_name],
                linewidth=1.8,
                markersize=3.6,
                label=_sign_norm_label(norm_name),
            )
        rho_ax.set_title("Spectral-radius diagnostics")
        rho_ax.set_ylabel("Spectral radius")
        rho_ax.legend(frameon=False, fontsize=8, ncol=2)
        scale_ax.set_title("Normalization scale factor")
        scale_ax.set_ylabel("Scale factor")

    for ax in axes.ravel():
        if not ax.axison:
            continue
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        ax.set_xlabel("Negative edge fraction")
        ax.grid(True, alpha=0.22)

    fig.suptitle("Sign-balance normalization ablation", y=0.985, fontsize=17)
    metric_handles = [
        Line2D([0], [0], color=metric_colors[metric], linewidth=2.2, label=metric)
        for metric in metrics
    ]
    norm_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=norm_styles[norm_name],
            marker=norm_markers[norm_name],
            linewidth=1.8,
            markersize=4,
            label=_sign_norm_label(norm_name),
        )
        for norm_name in norm_order
    ]
    fig.legend(
        metric_handles + norm_handles,
        [handle.get_label() for handle in metric_handles + norm_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=len(metric_handles + norm_handles),
        frameon=False,
    )
    fig.tight_layout(rect=[0.02, 0.025, 1.0, 0.90])
    out_path = _replace_path(out_path)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_sign_norm_ablation(combined: pd.DataFrame, out_dir: str, prefix: str = "sign_test_og_cel"):
    os.makedirs(out_dir, exist_ok=True)
    df = _prepare_sign_norm_ablation_df(combined, prefix)
    if df.empty:
        available = sorted(combined["mode"].astype(str).dropna().unique().tolist())
        preview = ", ".join(available[:16]) + (" ..." if len(available) > 16 else "")
        print(f"[warn] sign_norm_ablation: no rows matched prefix={prefix!r}.")
        print(f"[info] available modes: {preview}")
        return

    metrics = [m for m in ("MC", "IPC", "KR", "GR") if m in df.columns]
    if not metrics:
        print("[warn] sign_norm_ablation: no MC/IPC/KR/GR columns found.")
        return

    plot_norms = ("spectral_radius", "original_radius")
    norm_order = [
        n for n in plot_norms
        if n in set(df["normalization"].astype(str))
    ]
    if not norm_order:
        available = sorted(df["normalization"].astype(str).unique())
        print(
            "[warn] sign_norm_ablation: none of the requested plotted normalization modes "
            f"are present. Available: {available}"
        )
        return
    df = df[df["normalization"].astype(str).isin(norm_order)].copy()

    df = _assign_group_ids(df)
    df["unit_id"] = df["src"].astype(str) + ":" + df["group_id"].astype(str)

    metric_long = df.melt(
        id_vars=["normalization", "sign_frac", "unit_id"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    per_group_perf = (
        metric_long.groupby(["normalization", "sign_frac", "unit_id", "metric"], as_index=False)["value"]
        .mean()
    )
    perf_summary = _summarize_group_values(
        per_group_perf,
        "value",
        ["normalization", "sign_frac", "metric"],
    )
    perf_csv = _replace_path(os.path.join(out_dir, "sign_norm_ablation_mean_performance.csv"))
    perf_summary.to_csv(perf_csv, index=False)
    print(f"[saved] {perf_csv} (rows={len(perf_summary)})")

    hparam_keys = [
        "normalization",
        "sign_frac",
        "unit_id",
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
    ]
    hparam_agg = (
        df.groupby(hparam_keys, as_index=False)[metrics]
        .mean()
        .sort_values(hparam_keys)
        .reset_index(drop=True)
    )
    hparam_long = hparam_agg.melt(
        id_vars=["normalization", "sign_frac", "unit_id", "rho_target"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    per_target_cv = (
        hparam_long.groupby(
            ["normalization", "sign_frac", "unit_id", "rho_target", "metric"],
            as_index=False,
        )["value"]
        .agg(cv=lambda x: _dispersion(x.to_numpy()))
    )
    # Keep target radius out of the CV itself: calculate invariance across the
    # remaining hyperparameters at each target, then average targets per repeat.
    per_group_cv = (
        per_target_cv.groupby(
            ["normalization", "sign_frac", "unit_id", "metric"],
            as_index=False,
        )["cv"]
        .mean()
    )
    cv_summary = _summarize_group_values(
        per_group_cv,
        "cv",
        ["normalization", "sign_frac", "metric"],
    )
    cv_summary["n_rho_targets"] = int(df["rho_target"].nunique())
    cv_summary["cv_definition"] = "mean_within_target_cv"
    cv_csv = _replace_path(os.path.join(out_dir, "sign_norm_ablation_cv.csv"))
    cv_summary.to_csv(cv_csv, index=False)
    print(f"[saved] {cv_csv} (rows={len(cv_summary)})")

    diag_cols = [
        c for c in ("raw_rho", "ref_rho", "post_rho", "scale_factor")
        if c in df.columns
    ]
    scaling_summary = pd.DataFrame()
    if diag_cols:
        per_group_diag = (
            df.groupby(["normalization", "sign_frac", "unit_id"], as_index=False)[diag_cols]
            .mean()
        )
        parts = []
        for col in diag_cols:
            sub = (
                per_group_diag.groupby(["normalization", "sign_frac"], as_index=False)[col]
                .agg(**{col: "mean", f"{col}_std": "std", f"{col}_n": "count"})
            )
            sub[f"{col}_sem"] = sub[f"{col}_std"] / np.sqrt(sub[f"{col}_n"].clip(lower=1))
            parts.append(sub)
        scaling_summary = parts[0]
        for sub in parts[1:]:
            scaling_summary = scaling_summary.merge(sub, on=["normalization", "sign_frac"], how="outer")
        scaling_summary = scaling_summary.sort_values(["normalization", "sign_frac"]).reset_index(drop=True)
        fill_zero = {c: 0.0 for c in scaling_summary.columns if c.endswith(("_std", "_sem"))}
        scaling_summary = scaling_summary.fillna(fill_zero)
        scaling_csv = _replace_path(os.path.join(out_dir, "sign_norm_ablation_scaling.csv"))
        scaling_summary.to_csv(scaling_csv, index=False)
        print(f"[saved] {scaling_csv} (rows={len(scaling_summary)})")

    _save_sign_norm_combined_grid(
        perf_summary,
        cv_summary,
        scaling_summary,
        os.path.join(out_dir, "sign_norm_ablation_combined.png"),
        metric_order=metrics,
        norm_order=norm_order,
    )



# --------------------------- plots ---------------------------
def plot_frac_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    if not metrics:
        return

    mode_order = [
        "sign_test_og_cel0.0","sign_test_og_cel0.1","sign_test_og_cel0.2","sign_test_og_cel0.3","sign_test_og_cel0.4",
        "sign_test_og_cel0.5","sign_test_og_cel0.6","sign_test_og_cel0.7","sign_test_og_cel0.8","sign_test_og_cel0.9","sign_test_og_cel1.0",
    ]
    #mode_order = [
    #    "weight_test0.0","weight_test1.0","weight_test5.0","weight_test10.0","weight_test100.0",
    #    "weight_test1000.0", "weight_test10000.0",
    #]
    modes = [m for m in mode_order if m in set(disp["mode"].unique())]
    if not modes:
        print("fdfsa")
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
        x_label = "negative sign frac"
    elif all(str(m).startswith("weight_test") for m in modes):
        x_label = "weight_test value"
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






def plot_frac_cv_meanline(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    bins: int = 4,
    show: bool = True,
):
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
    highlight_frac = 0.06113256113256113
    highlight_atol = 1e-12
    plotted_any = False
    for idx, metric in enumerate(metrics):
        rows = []
        for mode in modes:
            frac = mode_values.get(mode, np.nan)
            y = mean_cv.get((mode, metric), np.nan)
            z = mean_perf.get((mode, metric), np.nan)
            if not (np.isfinite(frac) and np.isfinite(y) and np.isfinite(z)):
                continue
            rows.append((frac, y, z))
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        highlight_mask = np.isclose(xs, highlight_frac, rtol=0.0, atol=highlight_atol)
        if np.any(highlight_mask):
            ax.scatter(
                xs[highlight_mask],
                ys[highlight_mask],
                zs[highlight_mask],
                color="black",
                s=36,
                depthshade=False,
                zorder=6,
            )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("[warn] plot_frac_cv_meanline: no finite data to plot.")
        return

    if all(str(m).startswith("sign_test") for m in modes):
        x_label = "Negative Sign %"
    elif all(str(m).startswith("weight_test") for m in modes):
        x_label = "weight_test value"
    else:
        x_label = "mode value"

    # Use the full set of mode positions for x-limits.
    x_positions = [mode_values[m] for m in modes if np.isfinite(mode_values[m])]
    x_min, x_max = min(x_positions), max(x_positions)
    span = x_max - x_min
    pad = 0.05 * span if span > 0 else 1.0

    ax.set_xlabel(x_label, fontsize=18, labelpad=4)
    ax.set_ylabel("mean CV", fontsize=18, labelpad=2)
    ax.set_zlabel("mean Performance", fontsize=18, labelpad=2)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=bins))
    ax.set_xlim(x_min - pad, x_max + pad)
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "meanpoint_frac_cv_lines.png")
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(20, -25),
        back_view=(20, 145),
       dpi=600,
        tick_labelsize=16,
        tick_pad=-4,
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_weight_gauss_mean_cv(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
    local_sign_binary_csv: str = "",
):
    """
    3D line plot for weight_test (and other numeric modes):
      x = Gaussian magnitude (alpha) on log scale (log10 transform)
      y = mean CV (dispersion) per metric (MC/IPC/KR/GR)
      z = mean of an invariance column (prefers `kl_to_gaussian`)
    CV is not computed for the invariance column; we reuse the dispersion table for MC/IPC/KR/GR only.
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        return _mode_numeric_value(mode)

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    overlay = _load_local_sign_binary_overlay(local_sign_binary_csv)
    if not mode_vals:
        print("[warn] plot_weight_gauss_mean_cv: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])
    pos_logs = [np.log10(v) for _, v in mode_vals if v > 0]
    if not pos_logs:
        print("[warn] plot_weight_gauss_mean_cv: need at least one positive noise magnitude.")
        return
    has_zero = any(v == 0 for _, v in mode_vals)
    zero_mode = next((mode for mode, raw in mode_vals if raw == 0), None)
    x_zero = (min(pos_logs) - 0.60) if has_zero else np.nan

    def _x_plot(v: float) -> float:
        if v > 0:
            return float(np.log10(v))
        if v == 0 and has_zero:
            return float(x_zero)
        return float("nan")

    mean_tbl = _compute_mean_table(combined)
    inv_metric = "kl_to_gaussian"
    if inv_metric not in combined.columns:
        print(
            "[warn] plot_weight_gauss_mean_cv: required column 'kl_to_gaussian' not found. "
            "Regenerate combined CSVs from runs that include kl_to_gaussian."
        )
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
    z_label_base = "KL to Gaussian"
    cv_lookup = disp.groupby(["mode", "metric"])["dispersion"].mean()

    # Metrics that have both mean and cv
    metrics = [m for m in ("MC", "IPC", "KR", "GR") if (m in mean_tbl["metric"].unique()) and (m in disp["metric"].unique())]
    if not metrics:
        print("[warn] plot_weight_gauss_mean_cv: no overlapping metrics with mean+CV.")
        return

    # Choose a scaling for the invariance column so z-values stay in a readable range.
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
    x_line_vals = [_x_plot(v) for _, v in mode_vals]
    x_line_vals = [x for x in x_line_vals if np.isfinite(x)]
    x_min_line = float(np.min(x_line_vals))
    x_max_line = float(np.max(x_line_vals))
    z_vals = mean_mean_lookup.to_numpy(dtype=float) / z_scale
    z_min = float(np.nanmin(z_vals)) if z_vals.size else 0.0
    z_max = float(np.nanmax(z_vals)) if z_vals.size else 1.0
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        z_min, z_max = 0.0, 1.0
    if abs(z_max - z_min) < 1e-12:
        z_min -= 0.5
        z_max += 0.5

    for idx, metric in enumerate(metrics):
        rows = []
        for mode, x_raw in mode_vals:
            y_cv = cv_lookup.get((mode, metric), np.nan)
            z_mean = mean_mean_lookup.get(mode, np.nan)
            x_plot = _x_plot(x_raw)
            if not (np.isfinite(y_cv) and np.isfinite(z_mean) and np.isfinite(x_plot)):
                continue
            inv_scaled = z_mean / z_scale
            rows.append((x_plot, y_cv, inv_scaled))
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        # y-axis is metric value, z-axis is KL.
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        plotted = True

        z_lsb = cv_lookup.get((zero_mode, metric), np.nan) if zero_mode is not None else np.nan
        if not np.isfinite(z_lsb) and overlay is not None:
            z_lsb = overlay["cv_lookup"].get(("real", metric), np.nan)
        if np.isfinite(z_lsb):
            x_plane = np.array([[x_min_line, x_max_line], [x_min_line, x_max_line]], dtype=float)
            y_plane = np.full_like(x_plane, float(z_lsb))
            z_plane = np.array([[z_min, z_min], [z_max, z_max]], dtype=float)
            ax.plot_surface(
                x_plane,
                y_plane,
                z_plane,
                color=colors(idx % 10),
                alpha=0.12,
                shade=False,
                linewidth=0.0,
                antialiased=False,
            )
            ax.plot(
                [x_min_line, x_max_line, x_max_line, x_min_line, x_min_line],
                [z_lsb, z_lsb, z_lsb, z_lsb, z_lsb],
                [z_min, z_min, z_max, z_max, z_min],
                color=colors(idx % 10),
                linewidth=1.0,
                alpha=0.7,
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_mean_cv: no finite data to plot.")
        return

    ax.set_xlabel("log10(Noise Magnitude)", fontsize=18, labelpad=10)
    ax.set_ylabel("mean CV ", fontsize=18, labelpad=12)
    ax.set_zlabel(z_label_base, fontsize=18, labelpad=10)
    # helpful x ticks at common magnitudes if they are within range
    xticks = [v for v in (1, 10, 100, 1000) if np.isfinite(np.log10(v))]
    xtick_pos = [np.log10(v) for v in xticks]
    xtick_lbl = [str(v) for v in xticks]
    if has_zero:
        xtick_pos = [x_zero] + xtick_pos
        xtick_lbl = ["0"] + xtick_lbl
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl)
    x_pad = 0.08 * (x_max_line - x_min_line if x_max_line > x_min_line else 1.0)
    ax.set_xlim(x_min_line - x_pad, x_max_line + x_pad)
    #ax.legend()
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "weight_mean_cv_log3d.png")
    saved_paths = _save_3d_front_back(
        fig,
        ax,
        out_fig,
        front_view=(25, -15),
        back_view=(25, 15),
        dpi=600,
        tick_labelsize=16,
        tick_pad=-2,
        pane_fill=True,
        pane_facecolor=(0.95, 0.95, 0.95, 0.90),
        pane_edgecolor=(0.78, 0.78, 0.78, 1.0),
    )
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_weight_gauss_mean_perf(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
    local_sign_binary_csv: str = "",
):
    """
    3D line plot like plot_weight_gauss_mean_cv but using mean performance instead of mean CV:
      x = Gaussian magnitude (alpha) on log scale
      y = mean performance (per metric MC/IPC/KR/GR)
      z = invariance column (prefers `kl_to_gaussian`)
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        return _mode_numeric_value(mode)

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    overlay = _load_local_sign_binary_overlay(local_sign_binary_csv)
    if not mode_vals:
        print("[warn] plot_weight_gauss_mean_perf: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])
    pos_logs = [np.log10(v) for _, v in mode_vals if v > 0]
    if not pos_logs:
        print("[warn] plot_weight_gauss_mean_perf: need at least one positive noise magnitude.")
        return
    has_zero = any(v == 0 for _, v in mode_vals)
    x_zero = (min(pos_logs) - 0.60) if has_zero else np.nan

    def _x_plot(v: float) -> float:
        if v > 0:
            return float(np.log10(v))
        if v == 0 and has_zero:
            return float(x_zero)
        return float("nan")

    mean_tbl = _compute_mean_table(combined)
    mean_lookup = mean_tbl.groupby(["mode", "metric"])["mean"].mean()
    inv_metric = "kl_to_gaussian"
    if inv_metric not in combined.columns:
        print(
            "[warn] plot_weight_gauss_mean_perf: required column 'kl_to_gaussian' not found. "
            "Regenerate combined CSVs from runs that include kl_to_gaussian."
        )
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
    z_label_base = "KL to Gaussian"

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
    x_line_vals = [_x_plot(v) for _, v in mode_vals]
    x_line_vals = [x for x in x_line_vals if np.isfinite(x)]
    x_min_line = float(np.min(x_line_vals))
    x_max_line = float(np.max(x_line_vals))
    z_vals = mean_mean_lookup.to_numpy(dtype=float) / z_scale
    z_min = float(np.nanmin(z_vals)) if z_vals.size else 0.0
    z_max = float(np.nanmax(z_vals)) if z_vals.size else 1.0
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        z_min, z_max = 0.0, 1.0
    if abs(z_max - z_min) < 1e-12:
        z_min -= 0.5
        z_max += 0.5

    for idx, metric in enumerate(metrics):
        rows = []
        for mode, x_raw in mode_vals:
            y_mean = mean_lookup.get((mode, metric), np.nan)
            z_mean = mean_mean_lookup.get(mode, np.nan)
            x_plot = _x_plot(x_raw)
            if not (np.isfinite(y_mean) and np.isfinite(z_mean) and np.isfinite(x_plot)):
                continue
            inv_scaled = z_mean / z_scale
            rows.append((x_plot, y_mean, inv_scaled))
        if not rows:
            continue
        rows = sorted(rows, key=lambda t: t[0])
        xs, ys, zs = map(np.asarray, zip(*rows))
        # y-axis is metric value, z-axis is KL.
        ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10), label=metric)
        plotted = True

        if overlay is not None:
            z_lsb = overlay["mean_lookup"].get(("real", metric), np.nan)
            if has_zero:
                y0 = [r[1] for r in rows if abs(r[0] - x_zero) < 1e-12]
                if y0:
                    z_lsb = float(y0[0])
            if np.isfinite(z_lsb):
                x_plane = np.array([[x_min_line, x_max_line], [x_min_line, x_max_line]], dtype=float)
                y_plane = np.full_like(x_plane, float(z_lsb))
                z_plane = np.array([[z_min, z_min], [z_max, z_max]], dtype=float)
                ax.plot_surface(
                    x_plane,
                    y_plane,
                    z_plane,
                    color=colors(idx % 10),
                    alpha=0.12,
                    shade=False,
                    linewidth=0.0,
                    antialiased=False,
                )
                ax.plot(
                    [x_min_line, x_max_line, x_max_line, x_min_line, x_min_line],
                    [z_lsb, z_lsb, z_lsb, z_lsb, z_lsb],
                    [z_min, z_min, z_max, z_max, z_min],
                    color=colors(idx % 10),
                    linewidth=1.0,
                    alpha=0.7,
                )
                plotted = True

    if not plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_mean_perf: no finite data to plot.")
        return

    ax.set_xlabel("log10(Noise Magnitude)", fontsize=18, labelpad=10)
    ax.set_ylabel("Mean Metric Value", fontsize=18, labelpad=10)
    ax.set_zlabel(z_label_base, fontsize=18, labelpad=18)
    xticks = [v for v in (1, 10, 100, 1000) if np.isfinite(np.log10(v))]
    xtick_pos = [np.log10(v) for v in xticks]
    xtick_lbl = [str(v) for v in xticks]
    if has_zero:
        xtick_pos = [x_zero] + xtick_pos
        xtick_lbl = ["0"] + xtick_lbl
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl)
    x_pad = 0.08 * (x_max_line - x_min_line if x_max_line > x_min_line else 1.0)
    ax.set_xlim(x_min_line - x_pad, x_max_line + x_pad)
    _tight_layout_quiet(fig)

    out_fig = os.path.join(out_dir, "weight_mean_perf_log3d.png")
    #saved_paths = _save_3d_front_back(
    #    fig,
    #    ax,
    #    out_fig,
    #    front_view=(22, -15),
    #    back_view=(22, 35),
    #
    #    dpi=600,
    #    tick_labelsize=16,
    #    tick_pad=-2,
    #)
    if show:
        plt.show()
    plt.close(fig)
    for out_path in saved_paths:
        print(f"[saved] {out_path}")


def plot_weight_gauss_cv_metric_2d(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
):
    """
    2D weight-test plot for interpolation sweeps:
      x = interpolation fraction parsed from mode
      left y = mean CV across repeat groups
      right y = mean metric value across repeat groups
    """
    os.makedirs(out_dir, exist_ok=True)

    if disp.empty:
        print("[warn] plot_weight_gauss_cv_metric_2d: dispersion table is empty.")
        return

    mode_vals = []
    for mode in disp["mode"].dropna().unique():
        value = _mode_numeric_value(mode)
        if np.isfinite(value):
            mode_vals.append((mode, value))
    if not mode_vals:
        print("[warn] plot_weight_gauss_cv_metric_2d: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])

    mean_tbl = _compute_mean_table(combined)
    cv_lookup = disp.groupby(["mode", "metric"])["dispersion"].mean()
    mean_lookup = mean_tbl.groupby(["mode", "metric"])["mean"].mean()
    disp_metrics = set(disp["metric"].dropna().unique())
    mean_metrics = set(mean_tbl["metric"].dropna().unique())
    metrics = [m for m in ("MC", "IPC", "KR", "GR") if m in disp_metrics and m in mean_metrics]
    if not metrics:
        print("[warn] plot_weight_gauss_cv_metric_2d: no overlapping MC/IPC/KR/GR metrics found.")
        return

    zero_mode = next((mode for mode, raw in mode_vals if raw == 0), None)
    _write_weight_metric_difference_table(
        cv_lookup,
        "mean CV",
        out_dir,
        tex_name="weight_gauss_diff_mean_cv_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        log_prefix="plot_weight_gauss_cv_metric_2d",
        mode_value_symbol="f",
    )
    _write_weight_metric_difference_table(
        mean_lookup,
        "Mean Metric Value",
        out_dir,
        tex_name="weight_gauss_diff_mean_perf_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        log_prefix="plot_weight_gauss_cv_metric_2d",
        mode_value_symbol="f",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), dpi=300, sharex=True)
    colors = mpl.colormaps["tab10"]

    def _draw_panel(ax, lookup, y_label: str) -> bool:
        plotted = False
        for idx, metric in enumerate(metrics):
            rows = []
            for mode, x_val in mode_vals:
                y_val = lookup.get((mode, metric), np.nan)
                if np.isfinite(x_val) and np.isfinite(y_val):
                    rows.append((float(x_val), float(y_val)))
            if not rows:
                continue
            rows = sorted(rows, key=lambda t: t[0])
            xs, ys = map(np.asarray, zip(*rows))
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                color=colors(idx % 10),
                label=metric,
            )
            plotted = True
        ax.set_ylabel(y_label, fontsize=14)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        return plotted

    plotted_cv = _draw_panel(axes[0], cv_lookup, "mean CV")
    plotted_mean = _draw_panel(axes[1], mean_lookup, "Mean Metric Value")

    if not (plotted_cv or plotted_mean):
        plt.close(fig)
        print("[warn] plot_weight_gauss_cv_metric_2d: no finite data to plot.")
        return

    x_vals = [value for _, value in mode_vals if np.isfinite(value)]
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    x_pad = 0.06 * (x_max - x_min if x_max > x_min else 1.0)
    x_label = "Interpolation fraction"
    for ax in axes:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=min(6, max(2, len(set(x_vals))))))
        ax.set_xlabel(x_label, fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.92, wspace=0.14)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.025),
            ncol=len(handles),
            frameon=False,
            fontsize=10,
        )
    else:
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.97, wspace=0.14)

    out_png = os.path.join(out_dir, "weight_mean_cv_metric_2d.png")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_png}")


def plot_weight_gauss_perf_cv_grid(
    disp: pd.DataFrame,
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = True,
    local_sign_binary_csv: str = "",
):
    """
    Direct combined 2x2 figure (no intermediate perf/cv files):
      top row: mean performance (front/back)
      bottom:  mean CV (front/back)
      x-axis: raw Gaussian noise magnitude alpha
    """
    os.makedirs(out_dir, exist_ok=True)

    def mode_to_value(mode: str) -> float:
        return _mode_numeric_value(mode)

    mode_vals = []
    for m in combined["mode"].unique():
        v = mode_to_value(m)
        if np.isfinite(v):
            mode_vals.append((m, v))
    overlay = _load_local_sign_binary_overlay(local_sign_binary_csv)
    if not mode_vals:
        print("[warn] plot_weight_gauss_perf_cv_grid: no modes with numeric values found.")
        return
    mode_vals = sorted(mode_vals, key=lambda t: t[1])
    has_zero = any(v == 0 for _, v in mode_vals)
    x_zero = 0.0 if has_zero else np.nan

    def _x_plot(v: float) -> float:
        return float(v) if np.isfinite(v) else float("nan")

    mean_tbl = _compute_mean_table(combined)
    mean_lookup = mean_tbl.groupby(["mode", "metric"])["mean"].mean()
    cv_lookup = disp.groupby(["mode", "metric"])["dispersion"].mean()
    inv_metric = "kl_to_gaussian"
    if inv_metric not in combined.columns:
        print(
            "[warn] plot_weight_gauss_perf_cv_grid: required column 'kl_to_gaussian' not found. "
            "Regenerate combined CSVs from runs that include KL weight diagnostics."
        )
        return
    inv_tbl = _compute_mean_table(combined, metrics=[inv_metric])
    kl_lookup = (
        inv_tbl[inv_tbl["metric"] == inv_metric]
        .groupby("mode")["mean"]
        .mean()
    )
    z_label = "KL to Gaussian"
    print(f"[info] plot_weight_gauss_perf_cv_grid: using {inv_metric} for z-axis ({z_label}).")
    if kl_lookup.empty:
        print(f"[warn] plot_weight_gauss_perf_cv_grid: no values for invariance metric '{inv_metric}'.")
        return

    metrics = [m for m in ("MC", "IPC", "KR", "GR") if (m in mean_tbl["metric"].unique()) and (m in disp["metric"].unique())]
    if not metrics:
        print("[warn] plot_weight_gauss_perf_cv_grid: no overlapping metrics with mean+CV.")
        return

    max_kl = np.nanmax(kl_lookup.values) if len(kl_lookup) else np.nan
    if np.isfinite(max_kl) and max_kl > 0:
        z_power = int(np.floor(np.log10(max_kl)))
        z_scale = 10.0 ** z_power
    else:
        z_scale = 1.0

    colors = mpl.colormaps["tab10"]
    x_line_vals = [_x_plot(v) for _, v in mode_vals]
    x_line_vals = [x for x in x_line_vals if np.isfinite(x)]
    if not x_line_vals:
        print("[warn] plot_weight_gauss_perf_cv_grid: no finite x values to plot.")
        return
    x_min_line = float(np.min(x_line_vals))
    x_max_line = float(np.max(x_line_vals))
    z_vals = kl_lookup.to_numpy(dtype=float) / z_scale
    z_min = float(np.nanmin(z_vals)) if z_vals.size else 0.0
    z_max = float(np.nanmax(z_vals)) if z_vals.size else 1.0
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        z_min, z_max = 0.0, 1.0
    if abs(z_max - z_min) < 1e-12:
        z_min -= 0.5
        z_max += 0.5

    zero_mode = next((mode for mode, raw in mode_vals if raw == 0), None)

    def _draw_panel(ax, y_lookup, y_label: str):
        plotted = False
        y_lo_panel = np.inf
        y_hi_panel = -np.inf
        for idx, metric in enumerate(metrics):
            rows = []
            for mode, x_raw in mode_vals:
                y_val = y_lookup.get((mode, metric), np.nan)
                z_val = kl_lookup.get(mode, np.nan)
                x_val = _x_plot(x_raw)
                if not (np.isfinite(y_val) and np.isfinite(z_val) and np.isfinite(x_val)):
                    continue
                rows.append((x_val, y_val, z_val / z_scale))
            if not rows:
                continue
            rows = sorted(rows, key=lambda t: t[0])
            xs, ys, zs = map(np.asarray, zip(*rows))
            ax.plot(xs, ys, zs, marker="o", color=colors(idx % 10))
            plotted = True
            y_lo_panel = min(y_lo_panel, float(np.nanmin(ys)))
            y_hi_panel = max(y_hi_panel, float(np.nanmax(ys)))

            if overlay is not None:
                if y_label == "Mean Metric Value":
                    y_ref = overlay["mean_lookup"].get(("real", metric), np.nan)
                else:
                    y_ref = overlay["cv_lookup"].get(("real", metric), np.nan)
                if has_zero:
                    y0 = [r[1] for r in rows if abs(r[0] - x_zero) < 1e-12]
                    if y0:
                        y_ref = float(y0[0])
                if np.isfinite(y_ref):
                    x_plane = np.array([[x_min_line, x_max_line], [x_min_line, x_max_line]], dtype=float)
                    y_plane = np.full_like(x_plane, float(y_ref))
                    z_plane = np.array([[z_min, z_min], [z_max, z_max]], dtype=float)
                    ax.plot_surface(
                        x_plane,
                        y_plane,
                        z_plane,
                        color=colors(idx % 10),
                        alpha=0.05,
                        shade=False,
                        linewidth=0.0,
                        antialiased=False,
                    )
                    ax.plot(
                        [x_min_line, x_max_line, x_max_line, x_min_line, x_min_line],
                        [y_ref, y_ref, y_ref, y_ref, y_ref],
                        [z_min, z_min, z_max, z_max, z_min],
                        color=colors(idx % 10),
                        linewidth=1.0,
                        alpha=0.7,
                    )

        if not plotted:
            ax.set_axis_off()
            return False

        ax.set_xlabel("Noise Mag.", fontsize=18, labelpad=6)
        ax.set_ylabel(y_label, fontsize=18, labelpad=7)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(z_label, fontsize=18, labelpad=10, rotation=90)
        ax.zaxis.set_label_coords(1.06, 0.50)
        ax.xaxis.label.set_clip_on(False)
        ax.yaxis.label.set_clip_on(False)
        ax.zaxis.label.set_clip_on(False)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
        x_pad = 0.08 * (x_max_line - x_min_line if x_max_line > x_min_line else 1.0)
        ax.set_xlim(x_min_line - x_pad, x_max_line + x_pad)
        if np.isfinite(y_lo_panel) and np.isfinite(y_hi_panel):
            y_pad = 0.08 * (y_hi_panel - y_lo_panel if y_hi_panel > y_lo_panel else 1.0)
            ax.set_ylim(y_lo_panel - y_pad, y_hi_panel + y_pad)
        ax.set_zlim(z_min, z_max)
        _style_3d_axis(ax, tick_labelsize=18, tick_pad=-1)
        _clear_3d_axis_background(ax)
        return True

    _write_weight_metric_difference_table(
        cv_lookup,
        "mean CV",
        out_dir,
        tex_name="weight_gauss_diff_mean_cv_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        overlay_lookup=overlay["cv_lookup"] if overlay is not None else None,
        log_prefix="plot_weight_gauss_perf_cv_grid",
    )
    _write_weight_metric_difference_table(
        mean_lookup,
        "Mean Metric Value",
        out_dir,
        tex_name="weight_gauss_diff_mean_perf_table.tex",
        mode_vals=mode_vals,
        metrics=metrics,
        zero_mode=zero_mode,
        overlay_lookup=overlay["mean_lookup"] if overlay is not None else None,
        log_prefix="plot_weight_gauss_perf_cv_grid",
    )

    fig = plt.figure(figsize=(14.0, 10.0), dpi=300)
    ax00 = fig.add_subplot(2, 2, 1, projection="3d")
    ax01 = fig.add_subplot(2, 2, 2, projection="3d")
    ax10 = fig.add_subplot(2, 2, 3, projection="3d")
    ax11 = fig.add_subplot(2, 2, 4, projection="3d")

    any_plotted = False

    any_plotted |= _draw_panel(ax10, cv_lookup, "mean CV")
    any_plotted |= _draw_panel(ax11, cv_lookup, "mean CV")
    any_plotted |= _draw_panel(ax00, mean_lookup, "Mean Metric Value")
    any_plotted |= _draw_panel(ax01, mean_lookup, "Mean Metric Value")
    if not any_plotted:
        plt.close(fig)
        print("[warn] plot_weight_gauss_perf_cv_grid: no finite data to plot.")
        return

    front_view = (22, -20)
    back_view = (22, 20)
    for ax in (ax00, ax10):
        ax.view_init(elev=front_view[0], azim=front_view[1])
        ax.set_zorder(100)
    for ax in (ax01, ax11):
        ax.view_init(elev=back_view[0], azim=back_view[1])
        ax.set_zorder(2)

    # Bottom-row y endpoints can crowd x-corner ticks; prune only y endpoints.
    for ax in (ax10, ax11,ax00, ax01):
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="upper"))

    fig.subplots_adjust(left=-0.00, right=0.955, bottom=-0.0, top=1.05, wspace=0.2, hspace=-0.05)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "weight_mean_perf_cv_grid_3d.png")
    fig.savefig(
        out_png,
        dpi=600,
        facecolor="white",
        edgecolor="white",
    )
    if show:
        plt.show()
    plt.close(fig)
    print(f"[saved] {out_png}")


def plot_rho_cv_other_perf(
    combined: pd.DataFrame,
    out_dir: str,
    show: bool = False,
    model: str = "",
    drop_kr_gr: bool = False,
):
    """
    3D line plot at fixed spectral radius, one figure per mode:
      x = spectral radius (rho_target)
      y = CV across non-rho hyperparameters (leak, input_scale, neuron_bias) within each fixed rho
      z = mean performance at that fixed rho

    CV is computed per (mode, src, group_id, rho_target, metric) across leak/input_scale/neuron_bias,
    then averaged across groups for each (mode, rho_target, metric).
    """
    os.makedirs(out_dir, exist_ok=True)

    metric_cols = [m for m in ("MC", "IPC", "KR", "GR") if m in combined.columns]
    if drop_kr_gr:
        metric_cols = [m for m in metric_cols if m not in ("KR", "GR")]
    required_cols = ["mode", "src", "shuffle_id", "rho_target", "leak", "input_scale", "neuron_bias"] + metric_cols
    if not metric_cols:
        print("[warn] plot_rho_cv_other_perf: no selected metric columns found.")
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

    for c in ("rho_target", "leak", "input_scale", "neuron_bias", *metric_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["rho_target"])].copy()
    if df.empty:
        print("[warn] plot_rho_cv_other_perf: no finite rho_target values.")
        return
    df = df[df["rho_target"] > 0].copy()
    if df.empty:
        print("[warn] plot_rho_cv_other_perf: no positive rho_target values for log-scale plot.")
        return

    df = _assign_group_ids(df)
    df_agg = _aggregate_over_hparams(df, metric_cols)
    if df_agg.empty:
        print("[warn] plot_rho_cv_other_perf: no aggregated rows after deduplication.")
        return

    keys = ["mode", "src", "group_id", "rho_target", "leak", "input_scale", "neuron_bias"]
    df_long = df_agg.melt(id_vars=keys, value_vars=metric_cols, var_name="metric", value_name="value")

    per_group_rho = (
        df_long.groupby(["mode", "src", "group_id", "rho_target", "metric"], as_index=False)
        .agg(
            cv_other=("value", lambda x: _dispersion(x.to_numpy(dtype=float))),
            perf_mean=("value", "mean"),
            n_other_hparams=("value", "size"),
        )
    )

    if per_group_rho.empty:
        print("[warn] plot_rho_cv_other_perf: no data to plot.")
        return

    modes_to_plot = sorted(per_group_rho["mode"].astype(str).dropna().unique().tolist())
    if model:
        modes_to_plot = [model]
    if not modes_to_plot:
        print("[warn] plot_rho_cv_other_perf: no modes to plot.")
        return

    saved_any = False
    for mode_name in modes_to_plot:
        mode_rows = per_group_rho[per_group_rho["mode"].astype(str) == mode_name].copy()
        if mode_rows.empty:
            continue

        summary = (
            mode_rows.groupby(["rho_target", "metric"], as_index=False)
            .agg(
                cv_other=("cv_other", "mean"),
                perf_mean=("perf_mean", "mean"),
                n_groups=("group_id", "size"),
            )
        )
        if summary.empty:
            continue

        left_metrics = [m for m in ("MC", "IPC") if m in metric_cols]
        right_metrics = [m for m in ("GR", "KR") if m in metric_cols]
        panel_specs: list[tuple[str, list[str]]] = []
        if left_metrics:
            panel_specs.append(("MC + IPC", left_metrics))
        if right_metrics:
            panel_specs.append(("GR + KR", right_metrics))
        if not panel_specs:
            continue

        fig = plt.figure(figsize=(6.2 * len(panel_specs), 6), dpi=140)
        colors = mpl.colormaps["tab10"]
        metric_color = {
            "MC": colors(0),
            "IPC": colors(1),
            "KR": colors(2),
            "GR": colors(3),
        }
        rho_vals = sorted(v for v in summary["rho_target"].dropna().unique() if np.isfinite(v) and v > 0)
        tick_vals = []
        if rho_vals:
            preferred_ticks = [0.5, 1.0, 2.0, 4.0, 10.0]
            tick_vals = [v for v in preferred_ticks if any(np.isclose(v, rv) for rv in rho_vals)]
            if len(tick_vals) < min(4, len(rho_vals)):
                k = min(len(rho_vals), 6)
                idx = np.unique(np.linspace(0, len(rho_vals) - 1, k, dtype=int))
                tick_vals = [rho_vals[i] for i in idx]

        plotted_any = False
        for panel_idx, (_panel_title, panel_metrics) in enumerate(panel_specs, start=1):
            ax = fig.add_subplot(1, len(panel_specs), panel_idx, projection="3d")
            panel_plotted = False
            for metric in panel_metrics:
                sub = summary[summary["metric"] == metric].copy()
                if sub.empty:
                    continue
                sub = sub[np.isfinite(sub["rho_target"]) & np.isfinite(sub["cv_other"]) & np.isfinite(sub["perf_mean"])]
                if sub.empty:
                    continue
                sub = sub.sort_values("rho_target")
                xs = np.log10(sub["rho_target"].to_numpy(float))
                ys = sub["cv_other"].to_numpy(float)
                zs = sub["perf_mean"].to_numpy(float)
                ax.plot(xs, ys, zs, marker="o", color=metric_color.get(metric, colors(0)), label=metric)
                panel_plotted = True
                plotted_any = True

            if not panel_plotted:
                plt.delaxes(ax)
                continue

            if tick_vals:
                ax.set_xticks(np.log10(tick_vals))
                ax.set_xticklabels([f"{v}" for v in tick_vals])
            ax.set_xlabel("rho (log10)", fontsize=13, labelpad=2)
            ax.set_ylabel("CV(leak,input)", fontsize=11, labelpad=2)
            ax.set_zlabel("mean metric", fontsize=13, labelpad=3)
            # Match both panels to the left/front viewpoint.
            ax.view_init(elev=22, azim=-35)
            _style_3d_axis(ax, tick_labelsize=12, tick_pad=-1)
            # Requested: integer std ticks for GR/KR panel only; keep MC/IPC as floats.
            if "GR" in panel_metrics and "MC" not in panel_metrics and "IPC" not in panel_metrics:
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))
            else:
                ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2g"))

        if not plotted_any:
            plt.close(fig)
            continue

        fig.subplots_adjust(left=0.00, right=0.99, bottom=0.04, top=0.99, wspace=0.02)

        safe_mode = re.sub(r"[^A-Za-z0-9._-]+", "_", mode_name)
        out_name = f"rho_cv_other_perf_3d.{safe_mode}.png"
        out_fig = os.path.join(out_dir, out_name)
        fig.savefig(
            out_fig,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.005,
            facecolor="white",
            edgecolor="white",
        )
        if show:
            plt.show()
        plt.close(fig)
        print(f"[saved] {out_fig}")
        saved_any = True

    if not saved_any:
        print("[warn] plot_rho_cv_other_perf: no finite data to plot.")

def plot_overlaid_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int, mode_preset: str = "all"):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    present_modes = list(dict.fromkeys(disp["mode"].astype(str).dropna().tolist()))
    present_mode_set = set(present_modes)
    # Consistent ordering/colors to make panels easier to compare.
    mode_preset = str(mode_preset or "all").strip().lower()
    if ANALYSIS_MODE_FILTER:
        mode_order = [str(m) for m in ANALYSIS_MODE_FILTER if str(m) in present_mode_set]
        if "global_sign_pres" in present_mode_set and "global_sign_pres" not in mode_order:
            mode_order.append("global_sign_pres")
    elif mode_preset == "all_shuf":
        mode_order = (
                "global_sign_pres",
                "real",
                "global_sign_pres_real_w",
                "global_sign_pres_real_weight",
                "shuffle",
                "conn_shuf_only",
                "celW+connShuf",
                "local_sign+binary", ##this is localsign+signed binary weights
                "binary_base", ## unsigned binary
                "binary_base_topology_shuffle", ##unsigned binary w shuffle
                "binary+shuffle",
                "binary+conshuffle+wshuffle",
            )

        
    else:
        
        mode_order = [
            "real",
            "cel+randN",
            "er+randN",
            "ws_p0.1+randN",
            "local_sign",
            "local_sign+flat",
            "local_sign+binary",
            "global_sign_pres",
            "global_sign_pres_real_w",
            "global_sign_pres_real_weight",
            "binary_base",
        ]
    extras = [m for m in present_modes if m not in mode_order]
    modes = list(mode_order) + extras
    # Keep these exact colors per request.
    preserved_colors = {
        "real": "#32a2f2",
        "local_sign+binary": "#7488FF",
        "binary_base": "#78dee5",
    }
    # Colorblind-safe fallback palette for remaining modes.
    cb_palette = [
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#000000",  # black
    ]
    color_map = dict(preserved_colors)
    palette_idx = 0
    for mode_name in modes:
        if mode_name in color_map:
            continue
        color_map[mode_name] = cb_palette[palette_idx % len(cb_palette)]
        palette_idx += 1

    if not metrics:
        return
    plt.rcParams.update({
        "axes.titlesize": 25,
        "axes.labelsize": 22,
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
                display_mode = _short_legend_name(mode)
                ax.plot(
                    centers,
                    frac,
                    drawstyle="steps-mid",
                    linewidth=2.8,
                    alpha=0.9,
                    label=display_mode,
                    color=color,
                )
                # Keep the location marker tied to the same architecture color as its histogram.
                med = float(np.median(s))
                ax.axvline(
                    med,
                    color=color if color is not None else "#8a8a8a",
                    alpha=0.4,
                    linewidth=1.2,
                    linestyle="--",
                )
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
                        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        if legend_handles and legend_labels:
            leg = fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=3,
                frameon=True,
                fontsize=18,
                columnspacing=1.0,
                handlelength=1.8,
                handletextpad=0.5,
                borderaxespad=0.2,
            )
            leg.get_frame().set_alpha(0.9)
        fig.subplots_adjust(left=0.09, right=0.995, top=0.965, bottom=0.20, wspace=0.03, hspace=0.18)
        page = start // per_fig + 1
        suffix = "" if len(metrics) <= per_fig else f"_p{page}"
        name_root = "all_arch_hist_grid_all_shuf" if mode_preset == "all_shuf" else "all_arch_hist_grid"
        out_fig = _safe_path(os.path.join(out_dir, f"{name_root}{suffix}.png"))
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
            plt.scatter(
                sub_u["GR"],
                sub_u["MC"],
                s=36,
                alpha=0.5,
                label=_short_legend_name(cname),
                c=color,
            )
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
                    if tost['pval'].iloc[0]<0.06:
                        print(
                            rf"{m} & {_short_thesis_name(a)} vs {_short_thesis_name(b)} & "
                            rf"{tost['bound'].iloc[0]:.4g} & {tost['pval'].iloc[0]:.4g} \\"
                        )

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
        a_raw, _ = _read_compare_table(args.compare_mean_a, value_col=args.compare_value_col)
        b_raw, _ = _read_compare_table(args.compare_mean_b, value_col=args.compare_value_col)
        if ANALYSIS_MODE_FILTER:
            a_f = _filter_to_modes(a_raw, label="compare A")
            b_f = _filter_to_modes(b_raw, label="compare B")
            if a_f.empty:
                raise ValueError(
                    f"Mode filter removed all rows from compare A. Requested modes: {ANALYSIS_MODE_FILTER}"
                )
            if b_f.empty:
                raise ValueError(
                    f"Mode filter removed all rows from compare B. Requested modes: {ANALYSIS_MODE_FILTER}"
                )
            tmp_a = _safe_path(os.path.join(args.out_dir, "compare.filtered.A.csv"))
            tmp_b = _safe_path(os.path.join(args.out_dir, "compare.filtered.B.csv"))
            a_f.to_csv(tmp_a, index=False)
            b_f.to_csv(tmp_b, index=False)
            compare_a_path = tmp_a
            compare_b_path = tmp_b
        else:
            compare_a_path = args.compare_mean_a
            compare_b_path = args.compare_mean_b
        comp_tbl = compare_mode_metric_means(
            compare_a_path,
            compare_b_path,
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
            ref_tbl, ref_col = _read_compare_table(compare_a_path, value_col=args.compare_value_col)
            new_tbl, new_col = _read_compare_table(compare_b_path, value_col=args.compare_value_col)
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
    combined = _filter_to_modes(combined, label="combined")
    if combined.empty:
        raise ValueError(
            f"Mode filter removed all rows from combined input. Requested modes: {ANALYSIS_MODE_FILTER}"
        )

    if args.sign_norm_ablation:
        plot_sign_norm_ablation(combined, args.out_dir, prefix=args.sign_norm_prefix)
        return

    # Save a copy (non-destructive; versioned if exists)
    out_comb = _safe_path(os.path.join(args.out_dir, "combined.ALL.csv"))
    #combined.to_csv(out_comb, index=False)
    #print(f"[saved] {out_comb}  (rows={len(combined)})")

    # Compute and save dispersion table
    disp = _compute_dispersion_table(combined,mode="cv")
    #out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    #disp.to_csv(out_disp, index=False)

    # Plots
    #plot_frac_cv_meanline(disp, combined, args.out_dir, bins=args.frac_cv_bins)
    #plot_weight_gauss_mean_cv(disp, combined, args.out_dir,show=False)
    #if args.weight_gauss_cv_only:
    #    plot_weight_gauss_cv_metric_2d(disp, combined, args.out_dir, show=False)
    #else:
    #    plot_weight_gauss_perf_cv_grid(
    #        disp,
    #        combined,
    #        args.out_dir,
    #        show=False,
    #        local_sign_binary_csv=args.local_sign_binary_csv,
    #    )
    #plot_rho_cv_other_perf( combined, args.out_dir, show=True, model=args.model, drop_kr_gr=args.rho_cv_drop_kr_gr,)
    plot_overlaid_arch_histograms(disp, args.out_dir, args.bins)
    #plot_mc_vs_gr_all_arch(combined, args.out_dir, args.scatter_alpha)
    #print_kruskal_wallis_tables(disp)


if __name__ == "__main__":
    main()
