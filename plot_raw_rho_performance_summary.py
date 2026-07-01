#!/usr/bin/env python3
"""Combine shuffle controls and sign-flip sweeps as performance vs raw rho."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib as mpl

if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator
import numpy as np
import pandas as pd

from util.graph_utils import _assign_group_ids


METRICS = ("GR", "IPC", "KR", "MC")

SHORT_NAMES = {
    "real": "C. elegans",
    "shuffle": "Weight shuffle",
    "celW+connShuf": "Conn. + wt. shuffle",
    "conn_shuf_only": "Connection shuffle",
    "binary_base": "Binary wt.",
    "binary_base_topology_shuffle": "Binary wt. + shuf.",
    "local_sign+binary": "Sign-pres. pm1 wt.",
    "global_sign_pres": "Sign-pres. pm1 + wt. shuf.",
    "binary+shuffle": "pm1 + conn. shuffle",
    "binary+conshuffle+wshuffle": "pm1 sign-pres. conn. + wt. shuf.",
}

SHUFFLE_FAMILIES = [
    (
        "Binary controls",
        ["binary_base", "binary_base_topology_shuffle"],
        "#5f6b7a",
        "s",
    ),
    (
        "PM1 controls",
        [
            "local_sign+binary",
            "binary+shuffle",
            "global_sign_pres",
            "binary+conshuffle+wshuffle",
        ],
        "#009e73",
        "D",
    ),
    (
        "C. elegans shuffle controls",
        ["real", "shuffle", "celW+connShuf", "conn_shuf_only"],
        "#0072b2",
        "^",
    ),
]

SIGN_SWEEPS = [
    (
        "Matched C. elegans sweep",
        "matched_cel",
        "good_results/good_cel_new/matched_cel/combined.ALL.csv",
        "sign_test_og_cel",
        "#d55e00",
    ),
    (
        "Removed C. elegans sweep",
        "removed_cel",
        "good_results/good_cel_new/removed_cel/combined.ALL.csv",
        "sign_test_og_cel",
        "#e69f00",
    ),
    (
        "Matched ER sweep",
        "matched_er",
        "good_results/good_cel_new/matched_er/combined.ALL.csv",
        "sign_test_er",
        "#56b4e9",
    ),
]

NORM_STYLES = {
    "spectral_radius": ("-", "scale by $\\rho(W)$"),
}


def _read_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"mode", "src", "shuffle_id", "raw_rho", *METRICS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in ("raw_rho", *METRICS):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df["raw_rho"])].copy()
    return _assign_group_ids(df)


def _per_group_metric_summary(df: pd.DataFrame, extra_cols: list[str]) -> pd.DataFrame:
    group_cols = ["mode", "src", "group_id", *extra_cols]
    raw_rho = (
        df.groupby(group_cols, as_index=False)["raw_rho"]
        .mean()
        .rename(columns={"raw_rho": "raw_rho_group"})
    )
    long = df.melt(
        id_vars=group_cols,
        value_vars=list(METRICS),
        var_name="metric",
        value_name="performance",
    ).dropna(subset=["performance"])
    per_group = (
        long.groupby(group_cols + ["metric"], as_index=False)
        .agg(
            performance=("performance", "mean"),
            cv=("performance", lambda values: float(np.std(values) / (abs(np.mean(values)) + 1e-12))),
            n_hparams=("performance", "size"),
        )
        .merge(raw_rho, on=group_cols, how="left")
        .dropna(subset=["raw_rho_group"])
    )
    summary_cols = ["mode", *extra_cols, "metric"]
    summary = (
        per_group.groupby(summary_cols, as_index=False)
        .agg(
            raw_rho=("raw_rho_group", "mean"),
            raw_rho_sd=("raw_rho_group", "std"),
            performance=("performance", "mean"),
            performance_sd=("performance", "std"),
            cv=("cv", "mean"),
            cv_sd=("cv", "std"),
            n_groups=("cv", "size"),
            n_hparams=("n_hparams", "mean"),
        )
        .sort_values(summary_cols)
        .reset_index(drop=True)
    )
    summary["raw_rho_sd"] = summary["raw_rho_sd"].fillna(0.0)
    summary["performance_sd"] = summary["performance_sd"].fillna(0.0)
    summary["cv_sd"] = summary["cv_sd"].fillna(0.0)
    return summary


def _parse_sign_sweep_metadata(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    pat = re.compile(
        rf"^{re.escape(prefix)}([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"__norm_([A-Za-z0-9_+-]+)$"
    )
    parsed = out["mode"].astype(str).map(lambda mode: pat.match(mode))
    out["sign_frac"] = parsed.map(lambda m: float(m.group(1)) if m else np.nan)
    out["normalization"] = parsed.map(lambda m: m.group(2) if m else "")
    out = out[np.isfinite(out["sign_frac"]) & out["normalization"].isin(NORM_STYLES)].copy()
    return out


def build_summary(
    shuffle_csv: str | Path,
    sign_root: str | Path,
    max_sign_frac: float = 0.5,
) -> pd.DataFrame:
    frames = []

    shuffle = _read_results(shuffle_csv)
    shuffle["series_kind"] = "shuffle"
    shuffle_summary = _per_group_metric_summary(shuffle, ["series_kind"])
    for family, modes, color, marker in SHUFFLE_FAMILIES:
        fam = shuffle_summary[shuffle_summary["mode"].isin(modes)].copy()
        if fam.empty:
            continue
        fam["series"] = family
        fam["series_key"] = family
        fam["color"] = color
        fam["marker"] = marker
        fam["normalization"] = ""
        fam["sign_frac"] = np.nan
        frames.append(fam)

    sign_root = Path(sign_root)
    for label, dataset_key, rel_path, prefix, color in SIGN_SWEEPS:
        sign_path = sign_root / dataset_key / "combined.ALL.csv"
        if not sign_path.exists():
            sign_path = Path(rel_path)
        sign = _parse_sign_sweep_metadata(_read_results(sign_path), prefix)
        sign = sign[sign["sign_frac"] <= float(max_sign_frac)].copy()
        if sign.empty:
            continue
        sign["series_kind"] = "sign_sweep"
        sign["dataset"] = label
        sign_summary = _per_group_metric_summary(
            sign,
            ["series_kind", "dataset", "sign_frac", "normalization"],
        )
        sign_summary["series"] = label
        sign_summary["series_key"] = sign_summary["dataset"]
        sign_summary["color"] = color
        sign_summary["marker"] = "o"
        frames.append(sign_summary)

    if not frames:
        raise ValueError("No summary rows were produced from the provided result files.")

    summary = pd.concat(frames, ignore_index=True, sort=False)
    return summary


def _format_rho_tick(value: float, _pos: int) -> str:
    if value >= 10:
        return f"{value:g}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _nice_log_ticks(lo: float, hi: float) -> list[float]:
    candidates = np.asarray(
        [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100],
        dtype=float,
    )
    return [float(v) for v in candidates if lo <= v <= hi]


def plot_summary(
    summary: pd.DataFrame,
    out_dir: str | Path,
    stem: str,
    y_scale: str = "linear",
) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.65,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.35), dpi=300, sharex=True)
    panel_specs = [
        ("GR", "performance", "GR"),
        ("IPC", "performance", "IPC"),
        ("KR", "performance", "KR"),
        ("MC", "performance", "MC"),
        ("GR", "cv", ""),
        ("IPC", "cv", ""),
        ("KR", "cv", ""),
        ("MC", "cv", ""),
    ]

    panel_letters = tuple("ABCDEFGH")
    for panel_idx, (ax, (metric, value_col, title)) in enumerate(zip(axes.ravel(), panel_specs)):
        metric_df = summary[summary["metric"] == metric].copy()
        shuffle_df = metric_df[metric_df["series_kind"] == "shuffle"]
        sign_df = metric_df[metric_df["series_kind"] == "sign_sweep"]

        for family, modes, color, marker in SHUFFLE_FAMILIES:
            fam = shuffle_df[shuffle_df["series"] == family].copy()
            if fam.empty:
                continue
            mode_order = {mode: i for i, mode in enumerate(modes)}
            fam["mode_order"] = fam["mode"].map(mode_order).fillna(999)
            fam = fam.sort_values(["mode_order", "raw_rho"])
            ax.plot(
                fam["raw_rho"],
                fam[value_col],
                color=color,
                linewidth=1.55,
                marker=marker,
                markersize=4.9,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.95,
                alpha=0.94,
                zorder=3,
            )

        for dataset, sub in sign_df.groupby("dataset", sort=False):
            sub = sub.sort_values("sign_frac")
            color = str(sub["color"].iloc[0])
            ax.plot(
                sub["raw_rho"],
                sub[value_col],
                color=color,
                linestyle="-",
                linewidth=1.45,
                alpha=0.90,
                zorder=2,
            )
            ax.scatter(
                sub["raw_rho"],
                sub[value_col],
                color="white",
                marker="o",
                s=24,
                edgecolor=color,
                linewidth=0.90,
                zorder=4,
            )

        ax.set_title(title, fontweight="semibold", pad=5)
        ax.text(
            -0.048,
            1.035,
            panel_letters[panel_idx],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            clip_on=False,
        )
        ax.set_xscale("log")
        ax.set_yscale(y_scale if value_col == "performance" else "linear")
        ax.set_xlim(2.8, 110.0)
        ax.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 40, 70, 100]))
        ax.xaxis.set_major_formatter(FuncFormatter(_format_rho_tick))
        if y_scale == "log" and value_col == "performance":
            ax.yaxis.set_major_formatter(FuncFormatter(_format_rho_tick))
        ax.grid(True, axis="y", which="major", color="#d9d9d9", linewidth=0.42, alpha=0.52)
        ax.grid(True, axis="x", which="major", color="#e5e5e5", linewidth=0.34, alpha=0.38)
        if y_scale == "log" and value_col == "performance":
            ax.grid(True, which="minor", axis="y", color="#e8e8e8", linewidth=0.30, alpha=0.30)
        ax.tick_params(length=2.8, width=0.65, pad=1.4, direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.65)

        finite_y = metric_df[value_col].to_numpy(float)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size:
            y_hi = float(np.nanmax(finite_y))
            if y_scale == "log" and value_col == "performance":
                positive = finite_y[finite_y > 0]
                y_lo = float(np.nanmin(positive)) if positive.size else 1e-2
                y_min, y_max = y_lo * 0.82, y_hi * 1.18
                ax.set_ylim(y_min, y_max)
                ticks = _nice_log_ticks(y_min, y_max)
                if len(ticks) >= 2:
                    ax.yaxis.set_major_locator(FixedLocator(ticks))
                    ax.yaxis.set_major_formatter(FuncFormatter(_format_rho_tick))
                    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            else:
                y_lo = min(0.0, float(np.nanmin(finite_y)))
                pad = max(0.08 * (y_hi - y_lo), 0.5)
                if value_col == "cv":
                    pad = max(0.10 * (y_hi - y_lo), 0.025)
                ax.set_ylim(y_lo, y_hi + pad)

    fig.supxlabel(r"Raw spectral radius $\rho(W)$ (log scale)", fontsize=10.2, y=0.103)
    fig.text(0.018, 0.665, "Mean performance", rotation=90, va="center", ha="center", fontsize=10.2)
    fig.text(0.018, 0.335, "Coefficient of variation", rotation=90, va="center", ha="center", fontsize=10.2)

    family_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            linewidth=1.55,
            markersize=4.9,
            label=label,
        )
        for label, _modes, color, marker in SHUFFLE_FAMILIES
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markerfacecolor="#d9d9d9",
            markeredgecolor=color,
            linewidth=1.45,
            markersize=4.8,
            label=label,
        )
        for label, _key, _path, _prefix, color in SIGN_SWEEPS
    ]
    fig.legend(
        handles=family_handles + dataset_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.018),
        ncol=6,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.7,
        handletextpad=0.42,
    )

    fig.subplots_adjust(left=0.062, right=0.992, bottom=0.178, top=0.905, wspace=0.24, hspace=0.34)

    paths = []
    for ext, dpi in (("png", 450), ("pdf", 450)):
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=dpi, facecolor="white")
        paths.append(str(out_path))
        print(f"[saved] {out_path}")
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffle-csv", default="good_results/shuf/combined.ALL.csv")
    parser.add_argument("--sign-root", default="good_results/good_cel_new")
    parser.add_argument("--out-dir", default="good_results/summary")
    parser.add_argument("--stem", default="raw_rho_performance_summary")
    parser.add_argument("--max-sign-frac", type=float, default=0.5)
    parser.add_argument("--y-scale", choices=("log", "linear"), default="linear")
    parser.add_argument("--write-summary-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args.shuffle_csv, args.sign_root, max_sign_frac=args.max_sign_frac)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.write_summary_csv:
        csv_path = out_dir / f"{args.stem}.csv"
        summary.to_csv(csv_path, index=False)
        print(f"[saved] {csv_path}")
    plot_summary(summary, out_dir, args.stem, y_scale=args.y_scale)


if __name__ == "__main__":
    main()
