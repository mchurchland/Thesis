#!/usr/bin/env python3
"""Plot mean performance and CV against raw rho for the control families."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from util.graph_utils import _assign_group_ids


# GR is plotted separately because lower GR indicates better generalization,
# unlike the higher-is-better performance metrics in this figure.
METRICS = ("IPC", "KR", "MC")

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
        "C. elegans E/I sweep",
        "cel_matched",
        "final_results/sign_frac/cel_matched/combined.ALL.GRKR_erank.rank_updated.csv",
        "sign_test_og_cel",
        "#d55e00",
    ),
]

NORM_STYLES = {
    "spectral_radius": ("-", "scale by $\\rho(W)$"),
}

EXTENDED_ABSTRACT_COLORS = {
    "Binary controls": "#7A5195",
    "PM1 controls": "#D17C00",
    "C. elegans shuffle controls": "#007F5F",
    "C. elegans E/I sweep": "#E07A5F",
    "Matched C. elegans sweep": "#E07A5F",
    "Removed C. elegans sweep": "#4C78A8",
    "Matched ER sweep": "#8F6BB3",
}


def _display_color(label: str, default: str, color_scheme: str) -> str:
    if color_scheme == "extended-abstract":
        return EXTENDED_ABSTRACT_COLORS.get(label, default)
    if color_scheme != "thesis":
        raise ValueError("color_scheme must be 'thesis' or 'extended-abstract'.")
    return default


def _read_results(
    path: str | Path,
    metrics: tuple[str, ...] = METRICS,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"mode", "src", "shuffle_id", "raw_rho", *metrics}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in ("raw_rho", *metrics):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df["raw_rho"])].copy()
    return _assign_group_ids(df)


def _per_group_metric_summary(
    df: pd.DataFrame,
    extra_cols: list[str],
    metrics: tuple[str, ...] = METRICS,
) -> pd.DataFrame:
    group_cols = ["mode", "src", "group_id", *extra_cols]
    raw_rho = (
        df.groupby(group_cols, as_index=False)["raw_rho"]
        .mean()
        .rename(columns={"raw_rho": "raw_rho_group"})
    )
    long = df.melt(
        id_vars=group_cols,
        value_vars=list(metrics),
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
    suffixed_pat = re.compile(
        rf"^{re.escape(prefix)}([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"__norm_([A-Za-z0-9_+-]+)$"
    )
    plain_pat = re.compile(
        rf"^{re.escape(prefix)}([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
    )
    modes = out["mode"].astype(str)
    suffixed = modes.map(suffixed_pat.match)
    plain = modes.map(plain_pat.match)
    supplied_normalization = (
        out["normalization"].fillna("").astype(str)
        if "normalization" in out.columns
        else pd.Series("", index=out.index, dtype=str)
    )
    out["sign_frac"] = [
        float(suffix_match.group(1))
        if suffix_match
        else float(plain_match.group(1)) if plain_match else np.nan
        for suffix_match, plain_match in zip(suffixed, plain)
    ]
    out["normalization"] = [
        suffix_match.group(2) if suffix_match else supplied_normalization.loc[idx]
        for idx, suffix_match in suffixed.items()
    ]
    out = out[np.isfinite(out["sign_frac"]) & out["normalization"].isin(NORM_STYLES)].copy()
    return out


def build_summary(
    shuffle_csv: str | Path,
    sign_root: str | Path,
    max_sign_frac: float = 0.5,
    metrics: tuple[str, ...] = METRICS,
    sign_sweeps=SIGN_SWEEPS,
) -> pd.DataFrame:
    frames = []

    shuffle = _read_results(shuffle_csv, metrics=metrics)
    shuffle["series_kind"] = "shuffle"
    shuffle_summary = _per_group_metric_summary(
        shuffle,
        ["series_kind"],
        metrics=metrics,
    )
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
    for label, dataset_key, rel_path, prefix, color in sign_sweeps:
        sign_path = sign_root / dataset_key / "combined.ALL.GRKR_erank.rank_updated.csv"
        if not sign_path.exists():
            sign_path = Path(rel_path)
        sign = _parse_sign_sweep_metadata(
            _read_results(sign_path, metrics=metrics),
            prefix,
        )
        sign = sign[sign["sign_frac"] <= float(max_sign_frac)].copy()
        if sign.empty:
            continue
        sign["series_kind"] = "sign_sweep"
        sign["dataset"] = label
        sign_summary = _per_group_metric_summary(
            sign,
            ["series_kind", "dataset", "sign_frac", "normalization"],
            metrics=metrics,
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


def plot_summary(
    summary: pd.DataFrame,
    out_dir: str | Path,
    stem: str,
    y_scale: str = "linear",
    show: bool = True,
    metrics: tuple[str, ...] = METRICS,
    color_scheme: str = "thesis",
    sign_sweeps=SIGN_SWEEPS,
) -> list[str]:
    if color_scheme not in {"thesis", "extended-abstract"}:
        raise ValueError("color_scheme must be 'thesis' or 'extended-abstract'.")
    if show and "tkagg" not in mpl.get_backend().lower():
        try:
            plt.switch_backend("TkAgg")
        except ImportError as exc:
            raise RuntimeError(
                "Interactive display was requested, but no GUI backend is available. "
                "Install Tk support or run with --no-show."
            ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        len(metrics),
        2,
        figsize=(7.15, 6.85),
        dpi=300,
        sharex=True,
        squeeze=False,
    )
    rho_ticks = np.asarray([3.0, 10.0, 25.0, 50.0, 100.0])

    for row, metric in enumerate(metrics):
        metric_df = summary[summary["metric"] == metric].copy()
        shuffle_df = metric_df[metric_df["series_kind"] == "shuffle"]
        sign_df = metric_df[metric_df["series_kind"] == "sign_sweep"]

        for col, value_col in enumerate(("performance", "cv")):
            ax = axes[row, col]
            for family, modes, color, marker in SHUFFLE_FAMILIES:
                color = _display_color(family, color, color_scheme)
                fam = shuffle_df[shuffle_df["series"] == family].copy()
                if fam.empty:
                    continue
                mode_order = {mode: i for i, mode in enumerate(modes)}
                fam["mode_order"] = fam["mode"].map(mode_order).fillna(999)
                fam = fam.sort_values(["mode_order", "raw_rho"])
                ax.plot(
                    np.sqrt(fam["raw_rho"]),
                    fam[value_col],
                    color=color,
                    linewidth=1.55,
                    marker=marker,
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=0.95,
                    alpha=0.94,
                    zorder=3,
                )

            for _dataset, sub in sign_df.groupby("dataset", sort=False):
                sub = sub.sort_values("sign_frac")
                color = str(sub["color"].iloc[0])
                color = _display_color(str(_dataset), color, color_scheme)
                ax.plot(
                    np.sqrt(sub["raw_rho"]),
                    sub[value_col],
                    color=color,
                    linewidth=1.55,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=0.95,
                    alpha=0.92,
                    zorder=4,
                )

            panel_idx = row * 2 + col
            ax.text(
                0.975,
                0.955,
                chr(65 + panel_idx),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11.5,
                fontweight="bold",
                zorder=20,
            )
            if row == 0:
                ax.set_title(
                    "Mean performance" if col == 0 else "Mean CV",
                    fontweight="semibold",
                    pad=5,
                )
            if col == 0:
                ax.set_ylabel(metric, fontweight="bold", labelpad=6)

            ax.set_xlim(np.sqrt(2.8), np.sqrt(110.0))
            ax.set_xticks(np.sqrt(rho_ticks), labels=["3", "10", "25", "50", "100"])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.grid(True, color="#dddddd", linewidth=0.45, alpha=0.65)
            ax.tick_params(length=2.7, width=0.7, pad=1.5, direction="out")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            finite_values = metric_df[value_col].to_numpy(float)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size:
                value_hi = float(np.max(finite_values))
                if col == 0 and y_scale == "log":
                    positive = finite_values[finite_values > 0]
                    if positive.size:
                        ax.set_yscale("log")
                        ax.set_ylim(float(np.min(positive)) * 0.90, value_hi * 1.10)
                else:
                    pad = max(0.08 * value_hi, 0.01 if col else 0.05)
                    ax.set_ylim(0.0, value_hi + pad)

    fig.supxlabel(r"Raw spectral radius $\rho(W)$", y=0.170, fontsize=11.5)

    family_handles = [
        Line2D(
            [0],
            [0],
            color=_display_color(label, color, color_scheme),
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=_display_color(label, color, color_scheme),
            linewidth=1.75,
            markersize=4.7,
            label=label,
        )
        for label, _modes, color, marker in SHUFFLE_FAMILIES
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=_display_color(label, color, color_scheme),
            marker="o",
            markerfacecolor="white",
            markeredgecolor=_display_color(label, color, color_scheme),
            linewidth=1.85,
            markersize=4.9,
            label=label,
        )
        for label, _key, _path, _prefix, color in sign_sweeps
    ]
    fig.legend(
        handles=family_handles + dataset_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.025),
        ncol=3,
        frameon=False,
        columnspacing=0.90,
        handlelength=1.65,
        handletextpad=0.45,
        fontsize=10.8,
    )

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.245, top=0.955, wspace=0.24, hspace=0.34)

    paths = []
    for ext, dpi in (("png", 450), ("pdf", 450)):
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(
            out_path,
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
        paths.append(str(out_path))
        print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shuffle-csv",
        default="final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv",
    )
    parser.add_argument("--sign-root", default="final_results/sign_frac")
    parser.add_argument("--out-dir", default="final_results/graphs/thesis/raw_rho")
    parser.add_argument("--stem", default="raw_rho_performance_summary")
    parser.add_argument("--max-sign-frac", type=float, default=0.5)
    parser.add_argument("--y-scale", choices=("log", "linear"), default="linear")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the interactive Matplotlib window (default: true)",
    )
    parser.add_argument("--write-summary-csv", action="store_true")
    parser.add_argument(
        "--color-scheme",
        choices=("thesis", "extended-abstract"),
        default="thesis",
    )
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
    plot_summary(
        summary,
        out_dir,
        args.stem,
        y_scale=args.y_scale,
        show=args.show,
        color_scheme=args.color_scheme,
    )


if __name__ == "__main__":
    main()
