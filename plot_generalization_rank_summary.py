#!/usr/bin/env python3
"""Create the standalone GR synthesis figure from rank-updated erank results.

GR is kept separate from IPC, KR, and MC because lower GR indicates better
generalization. The figure therefore places mean GR against mean CV using the
same direction on both axes: moving up/right means worse generalization and
greater hyperparameter sensitivity.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from util.graph_utils import _assign_group_ids


MAIN_CSV = Path("final_results/main/combined.ALL.GRKR_erank.rank_updated.csv")
SHUFFLE_CSV = Path("final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv")
SIGN_SPECS = (
    (
        "Matched C. elegans",
        Path("final_results/sign_frac/cel_matched/combined.ALL.GRKR_erank.rank_updated.csv"),
        "#d55e00",
    ),
    (
        "Predicted-polarity only",
        Path("final_results/sign_frac/cel_removed/combined.ALL.GRKR_erank.rank_updated.csv"),
        "#e69f00",
    ),
    (
        "Matched ER",
        Path("final_results/sign_frac/matched_er/combined.ALL.GRKR_erank.rank_updated.csv"),
        "#56b4e9",
    ),
)

CE_SIGN_FRACTION = 0.2425287356321839

MAIN_ORDER = (
    "real",
    "cel+randN",
    "er+randN",
    "ws_p0.1+randN",
    "local_sign",
    "local_sign+flat",
    "local_sign+binary",
    "global_sign_pres_real_weight",
    "binary_base",
)

SHORT_NAMES = {
    "real": "C. elegans",
    "cel+randN": "CE Gaussian",
    "er+randN": "ER Gaussian",
    "ws_p0.1+randN": "WS Gaussian",
    "local_sign": "Sign-pres. Gaussian",
    "local_sign+flat": "Sign-pres. uniform",
    "local_sign+binary": "Sign-pres. pm1",
    "global_sign_pres_real_weight": "Sign shuffle",
    "shuffle": "Weight shuffle",
    "celW+connShuf": "Conn. + wt. shuffle",
    "conn_shuf_only": "Connection shuffle",
    "global_sign_pres": "pm1 + wt. shuffle",
    "binary+shuffle": "pm1 + conn. shuffle",
    "binary+conshuffle+wshuffle": "pm1 sign/conn/wt shuffle",
    "binary_base": "Binary",
    "binary_base_topology_shuffle": "Binary shuffle",
}

PANEL_NAMES = {
    "real": "C. elegans",
    "shuffle": "Wt.",
    "celW+connShuf": "Conn. + wt.",
    "conn_shuf_only": "Conn.",
    "local_sign+binary": "PM1 baseline",
    "binary+shuffle": "Conn.",
    "global_sign_pres": "Wt.",
    "binary+conshuffle+wshuffle": "Conn. + wt.",
    "binary_base": "Binary",
    "binary_base_topology_shuffle": "Binary shuffle",
}

MARKERS = {
    "real": "*",
    "cel+randN": "o",
    "er+randN": "s",
    "ws_p0.1+randN": "^",
    "local_sign": "D",
    "local_sign+flat": "P",
    "local_sign+binary": "X",
    "global_sign_pres_real_weight": "<",
    "binary_base": ">",
    "shuffle": "o",
    "celW+connShuf": "s",
    "conn_shuf_only": "^",
    "global_sign_pres": "P",
    "binary+shuffle": "D",
    "binary+conshuffle+wshuffle": "X",
    "binary_base_topology_shuffle": "s",
}

MAIN_COLORS = {
    "real": "#b72fe3",
    "cel+randN": "#f2b705",
    "er+randN": "#f2b705",
    "ws_p0.1+randN": "#f2b705",
    "local_sign": "#b72fe3",
    "local_sign+flat": "#b72fe3",
    "local_sign+binary": "#b72fe3",
    "global_sign_pres_real_weight": "#b72fe3",
    "binary_base": "#1557d4",
}

SHUFFLE_FAMILIES = (
    (
        "C. elegans shuffles",
        "real",
        ("real", "shuffle", "celW+connShuf", "conn_shuf_only"),
        "#0072b2",
    ),
    (
        "PM1 shuffles",
        "local_sign+binary",
        (
            "local_sign+binary",
            "binary+shuffle",
            "global_sign_pres",
            "binary+conshuffle+wshuffle",
        ),
        "#009e73",
    ),
    (
        "Binary shuffles",
        "binary_base",
        ("binary_base", "binary_base_topology_shuffle"),
        "#5f6b7a",
    ),
)


def _read_grouped_gr(path: Path) -> pd.DataFrame:
    if "GRKR_erank.rank_updated" not in path.name:
        raise ValueError(f"GR input must be a rank-updated _erank file: {path}")
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {
        "mode",
        "src",
        "shuffle_id",
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
        "GR",
        "raw_rho",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = _assign_group_ids(df)
    df["GR"] = pd.to_numeric(df["GR"], errors="coerce")
    df["raw_rho"] = pd.to_numeric(df["raw_rho"], errors="coerce")
    df = df[np.isfinite(df["GR"]) & np.isfinite(df["raw_rho"])].copy()

    group_keys = ["mode", "src", "group_id"]
    hparam_keys = ["rho_target", "leak", "input_scale", "neuron_bias"]
    unique_hparams = (
        df.groupby(group_keys + hparam_keys, as_index=False)
        .agg(GR=("GR", "mean"), raw_rho=("raw_rho", "mean"))
    )

    def coefficient_of_variation(values: pd.Series) -> float:
        array = values.to_numpy(dtype=float)
        return float(np.std(array) / (abs(np.mean(array)) + 1e-12))

    grouped = (
        unique_hparams.groupby(group_keys, as_index=False)
        .agg(
            mean_gr=("GR", "mean"),
            gr_cv=("GR", coefficient_of_variation),
            raw_rho=("raw_rho", "mean"),
            n_hparams=("GR", "size"),
        )
    )
    grouped["source_file"] = str(path)
    return grouped


def _condition_summary(
    grouped: pd.DataFrame,
    context: str,
    series: str,
    sign_fraction: bool = False,
) -> pd.DataFrame:
    summary = (
        grouped.groupby("mode", as_index=False)
        .agg(
            mean_gr=("mean_gr", "mean"),
            gr_cv=("gr_cv", "mean"),
            raw_rho=("raw_rho", "mean"),
            n_groups=("mean_gr", "size"),
            source_file=("source_file", "first"),
        )
    )
    summary["context"] = context
    summary["series"] = series
    summary["sign_fraction"] = np.nan
    if sign_fraction:
        summary["sign_fraction"] = pd.to_numeric(
            summary["mode"].str.extract(
                r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
            )[0],
            errors="coerce",
        )
    return summary[
        [
            "context",
            "series",
            "mode",
            "sign_fraction",
            "mean_gr",
            "gr_cv",
            "raw_rho",
            "n_groups",
            "source_file",
        ]
    ]


def _draw_hdi_contour(
    ax,
    sub: pd.DataFrame,
    color,
    marker: str,
    *,
    contour_mass: float = 0.50,
    baseline: bool = False,
) -> tuple[float, float] | None:
    x = sub["gr_cv"].to_numpy(dtype=float)
    y = sub["mean_gr"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return None

    x_pad = max(0.06 * float(np.ptp(x)), 0.004)
    y_pad = max(0.06 * float(np.ptp(y)), 0.06)
    x_edges = np.linspace(float(np.min(x)) - x_pad, float(np.max(x)) + x_pad, 37)
    y_edges = np.linspace(float(np.min(y)) - y_pad, float(np.max(y)) + y_pad, 37)
    hist, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    density = gaussian_filter(hist, sigma=1.35, mode="constant")
    total = float(density.sum())
    if total > 0:
        ordered = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(ordered) / total
        level_idx = min(
            int(np.searchsorted(cumulative, contour_mass)),
            len(ordered) - 1,
        )
        level = float(ordered[level_idx])
        if level > 0:
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
            if baseline:
                ax.contour(
                    grid_x,
                    grid_y,
                    density,
                    levels=[level],
                    colors=["white"],
                    linewidths=3.0,
                    zorder=7,
                )
            ax.contour(
                grid_x,
                grid_y,
                density,
                levels=[level],
                colors=[color],
                linewidths=1.8 if baseline else 1.05,
                alpha=1.0 if baseline else 0.88,
                zorder=8 if baseline else 4,
            )

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    ax.scatter(
        [mean_x],
        [mean_y],
        s=74 if marker == "*" else 34,
        color=color,
        marker=marker,
        edgecolor="#202020",
        linewidth=0.65,
        zorder=9,
    )
    return mean_x, mean_y


def _style_gr_axis(ax, *, xlabel: bool = True, ylabel: bool = True) -> None:
    if xlabel:
        ax.set_xlabel("Mean CV")
    if ylabel:
        ax.set_ylabel("Mean GR")
    ax.grid(True, color="#d8d8d8", linewidth=0.5, alpha=0.62)
    ax.tick_params(length=3.0, width=0.7, pad=2.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_label(ax, label: str) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14.0,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _plot_main_architectures(ax, grouped: pd.DataFrame) -> list[Line2D]:
    handles: list[Line2D] = []
    available = set(grouped["mode"])
    for mode in MAIN_ORDER:
        if mode not in available:
            continue
        color = MAIN_COLORS[mode]
        marker = MARKERS[mode]
        _draw_hdi_contour(
            ax,
            grouped[grouped["mode"] == mode],
            color,
            marker,
            baseline=mode == "real",
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linewidth=1.5,
                markersize=7.5 if marker == "*" else 5.5,
                markeredgecolor="#202020",
                markeredgewidth=0.6,
                label=SHORT_NAMES[mode],
            )
        )
    _style_gr_axis(ax)
    _panel_label(ax, "A")
    return handles


def _plot_sign_sweeps(ax, summaries: list[tuple[str, pd.DataFrame, str]]) -> None:
    empirical_labeled = False
    for dataset, summary, color in summaries:
        ordered = summary.sort_values("sign_fraction")
        correlation = ordered["mean_gr"].corr(ordered["gr_cv"])
        ax.plot(
            ordered["sign_fraction"],
            ordered["gr_cv"],
            ordered["mean_gr"],
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=f"{dataset} ($r$={correlation:.2f})",
        )
        if dataset != "Matched ER":
            empirical = ordered.iloc[
                np.abs(ordered["sign_fraction"] - CE_SIGN_FRACTION).argmin()
            ]
            ax.scatter(
                [empirical["sign_fraction"]],
                [empirical["gr_cv"]],
                [empirical["mean_gr"]],
                color="black",
                s=30,
                depthshade=False,
                zorder=6,
                label=(
                    "C. elegans E/I edge balance"
                    if not empirical_labeled
                    else "_nolegend_"
                ),
            )
            empirical_labeled = True
    ax.set_xlabel("E/I edge balance", labelpad=1)
    ax.set_ylabel("Mean CV", labelpad=1)
    ax.set_zlabel("Mean GR", labelpad=2)
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.18, 1.0, 0.78), zoom=1.30)
    ax.view_init(elev=20, azim=-18)
    ax.tick_params(labelsize=8.4, pad=-1)
    ax.grid(True)
    ax.legend(
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="0.86",
        framealpha=1.0,
        fontsize=7.5,
        loc="upper left",
        bbox_to_anchor=(-0.11, 1.01),
        borderaxespad=0.0,
        handlelength=1.5,
    )
    _panel_label(ax, "B")


def _raw_rho_series(
    shuffle_summary: pd.DataFrame,
    sign_summaries: list[tuple[str, pd.DataFrame, str]],
) -> list[tuple[str, pd.DataFrame, str, str]]:
    series: list[tuple[str, pd.DataFrame, str, str]] = []
    for title, _baseline, modes, color in SHUFFLE_FAMILIES:
        mode_order = {mode: idx for idx, mode in enumerate(modes)}
        sub = shuffle_summary[shuffle_summary["mode"].isin(modes)].copy()
        sub["order"] = sub["mode"].map(mode_order)
        series.append((title, sub.sort_values("order"), color, "s"))
    for dataset, summary, color in sign_summaries:
        sub = summary[summary["sign_fraction"] <= 0.5].sort_values("sign_fraction")
        series.append((dataset, sub, color, "o"))
    return series


def _plot_raw_rho(
    ax_gr,
    ax_cv,
    series: list[tuple[str, pd.DataFrame, str, str]],
) -> None:
    for label, sub, color, marker in series:
        if sub.empty:
            continue
        for ax, column in ((ax_gr, "mean_gr"), (ax_cv, "gr_cv")):
            ax.plot(
                sub["raw_rho"],
                sub[column],
                color=color,
                linewidth=1.3,
                marker=marker,
                markersize=3.4,
                markerfacecolor="white",
                markeredgewidth=0.8,
                alpha=0.94,
                label=label,
            )
    for ax in (ax_gr, ax_cv):
        ax.set_xscale("log")
        ax.set_xlim(2.8, 110.0)
        ax.set_xticks([3, 10, 25, 50, 100], labels=["3", "10", "25", "50", "100"])
        ax.grid(True, color="#d8d8d8", linewidth=0.45, alpha=0.62)
        ax.tick_params(length=2.7, width=0.65, labelsize=8.6, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_gr.set_xlabel(r"Raw spectral radius $\rho(W)$", fontsize=10.2)
    ax_gr.set_ylabel("Mean GR", fontsize=10.2)
    ax_cv.set_ylabel("Mean CV", fontsize=10.2)
    ax_cv.set_xlabel(r"Raw spectral radius $\rho(W)$", fontsize=10.2)
    ax_gr.legend(
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="upper right",
        columnspacing=0.7,
        handlelength=1.35,
        handletextpad=0.35,
    )
    ax_cv.legend(
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="upper right",
        columnspacing=0.7,
        handlelength=1.35,
        handletextpad=0.35,
    )
    _panel_label(ax_gr, "C")
    _panel_label(ax_cv, "D")


def _plot_shuffle_family(
    ax,
    grouped: pd.DataFrame,
    title: str,
    baseline: str,
    modes: tuple[str, ...],
    color_norm,
    color_map,
    panel_label: str,
    common_limits: tuple[float, float, float, float],
) -> None:
    mode_raw_rho = grouped.groupby("mode")["raw_rho"].mean()
    baseline_rho = float(mode_raw_rho[baseline])
    legend_handles: list[Line2D] = []
    for mode in modes:
        sub = grouped[grouped["mode"] == mode]
        if sub.empty:
            continue
        delta = 100.0 * (float(mode_raw_rho[mode]) - baseline_rho) / max(abs(baseline_rho), 1e-12)
        color = color_map(color_norm(delta))
        point = _draw_hdi_contour(
            ax,
            sub,
            color,
            MARKERS.get(mode, "o"),
            baseline=mode == baseline,
        )
        if point is None:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=MARKERS.get(mode, "o"),
                linewidth=1.25,
                markersize=4.7,
                markeredgecolor="#202020",
                markeredgewidth=0.55,
                label=PANEL_NAMES.get(mode, SHORT_NAMES.get(mode, mode)),
            )
        )
    ax.set_xlim(common_limits[0], common_limits[1])
    ax.set_ylim(common_limits[2], common_limits[3])
    ax.set_title(title, fontweight="semibold")
    _style_gr_axis(ax)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=7.4,
        ncol=2 if len(legend_handles) > 2 else 1,
        columnspacing=0.65,
        handlelength=1.25,
        handletextpad=0.35,
    )
    _panel_label(ax, panel_label)


def create_figure(
    main_csv: Path,
    shuffle_csv: Path,
    sign_specs: tuple[tuple[str, Path, str], ...],
    out_dir: Path,
    stem: str,
) -> list[Path]:
    main_groups = _read_grouped_gr(main_csv)
    shuffle_groups = _read_grouped_gr(shuffle_csv)
    sign_groups = [
        (dataset, _read_grouped_gr(path), color, path)
        for dataset, path, color in sign_specs
    ]

    main_summary = _condition_summary(main_groups, "main", "Architecture comparison")
    shuffle_summary = _condition_summary(shuffle_groups, "shuffle", "Shuffle controls")
    sign_summaries: list[tuple[str, pd.DataFrame, str]] = []
    output_summaries = [main_summary, shuffle_summary]
    for dataset, grouped, color, _path in sign_groups:
        summary = _condition_summary(grouped, "sign_sweep", dataset, sign_fraction=True)
        sign_summaries.append((dataset, summary, color))
        output_summaries.append(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / f"{stem}.csv"
    pd.concat(output_summaries, ignore_index=True).to_csv(summary_csv, index=False)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.6,
            "axes.labelsize": 10.2,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(11.4, 12.2), dpi=300)
    outer = fig.add_gridspec(
        4,
        4,
        left=0.080,
        right=0.980,
        bottom=0.078,
        top=0.985,
        wspace=0.62,
        hspace=0.38,
        height_ratios=(1.15, 0.90, 1.0, 1.0),
    )
    ax_main = fig.add_subplot(outer[0, 0:2])
    ax_sign = fig.add_subplot(outer[0, 2:4], projection="3d")
    ax_raw_gr = fig.add_subplot(outer[1, 0:2])
    ax_raw_cv = fig.add_subplot(outer[1, 2:4])
    shuffle_axes = [
        fig.add_subplot(outer[2, 0:2]),
        fig.add_subplot(outer[2, 2:4]),
        fig.add_subplot(outer[3, 1:3]),
    ]

    main_handles = _plot_main_architectures(ax_main, main_groups)
    _plot_sign_sweeps(ax_sign, sign_summaries)
    _plot_raw_rho(
        ax_raw_gr,
        ax_raw_cv,
        _raw_rho_series(shuffle_summary, sign_summaries),
    )

    all_shuffle = shuffle_groups[
        shuffle_groups["mode"].isin(
            {mode for _title, _baseline, modes, _color in SHUFFLE_FAMILIES for mode in modes}
        )
    ]
    x_values = all_shuffle["gr_cv"].to_numpy(dtype=float)
    y_values = all_shuffle["mean_gr"].to_numpy(dtype=float)
    x_lo, x_hi = np.quantile(x_values, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y_values, [0.01, 0.99])
    x_pad = 0.05 * (x_hi - x_lo)
    y_pad = 0.05 * (y_hi - y_lo)
    common_limits = (x_lo - x_pad, x_hi + x_pad, y_lo - y_pad, y_hi + y_pad)

    mean_rho = shuffle_groups.groupby("mode")["raw_rho"].mean()
    rho_deltas = []
    for _title, baseline, modes, _color in SHUFFLE_FAMILIES:
        baseline_rho = float(mean_rho[baseline])
        for mode in modes:
            rho_deltas.append(
                100.0 * (float(mean_rho[mode]) - baseline_rho) / max(abs(baseline_rho), 1e-12)
            )
    max_abs_delta = max(abs(float(value)) for value in rho_deltas)
    rho_norm = mpl.colors.TwoSlopeNorm(
        vmin=-max_abs_delta,
        vcenter=0.0,
        vmax=max_abs_delta,
    )
    rho_cmap = mpl.colormaps["coolwarm"]

    for ax, family, panel_label in zip(shuffle_axes, SHUFFLE_FAMILIES, ("E", "F", "G")):
        title, baseline, modes, _color = family
        _plot_shuffle_family(
            ax,
            shuffle_groups,
            title,
            baseline,
            modes,
            rho_norm,
            rho_cmap,
            panel_label,
            common_limits,
        )

    ax_main.legend(
        handles=main_handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=7.8,
        columnspacing=0.75,
        handlelength=1.25,
        handletextpad=0.32,
    )
    sign_balance_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "architecture_sign_balance",
        [
            (0.0, MAIN_COLORS["binary_base"]),
            (CE_SIGN_FRACTION / 0.5, MAIN_COLORS["real"]),
            (1.0, MAIN_COLORS["cel+randN"]),
        ],
    )
    sign_balance_scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=0.5),
        cmap=sign_balance_cmap,
    )
    sign_balance_scalar.set_array([])
    main_position = ax_main.get_position()
    main_color_ax = fig.add_axes(
        [
            main_position.x1 + 0.008,
            main_position.y0 + 0.08 * main_position.height,
            0.014,
            0.74 * main_position.height,
        ]
    )
    main_colorbar = fig.colorbar(
        sign_balance_scalar,
        cax=main_color_ax,
        orientation="vertical",
    )
    main_colorbar.set_ticks([0.0, CE_SIGN_FRACTION, 0.5])
    main_colorbar.set_ticklabels(["0%", "CE 24.3%", "50%"])
    main_colorbar.set_label(
        "E/I edge balance",
        fontsize=9.0,
        rotation=270,
        labelpad=11,
    )
    main_colorbar.ax.tick_params(labelsize=8.2, length=2.8, pad=1.4)
    main_colorbar.outline.set_linewidth(0.45)

    scalar = mpl.cm.ScalarMappable(norm=rho_norm, cmap=rho_cmap)
    scalar.set_array([])
    color_ax = fig.add_axes([0.285, 0.026, 0.430, 0.008])
    colorbar = fig.colorbar(scalar, cax=color_ax, orientation="horizontal")
    colorbar.set_label(
        r"Panels E--G only: shuffle-control $\Delta\rho_{\mathrm{raw}}$ from row baseline (%)",
        fontsize=9.2,
        labelpad=2,
    )
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=8.2, length=2.4, pad=1.2)
    colorbar.outline.set_linewidth(0.45)

    output_paths: list[Path] = [summary_csv]
    for extension, dpi in (("png", 450), ("pdf", 450)):
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        output_paths.append(path)
        print(f"[saved] {path}")
    print(f"[saved] {summary_csv}")
    plt.close(fig)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=Path, default=MAIN_CSV)
    parser.add_argument("--shuffle-csv", type=Path, default=SHUFFLE_CSV)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("final_results/graphs/thesis/generalization_rank"),
    )
    parser.add_argument("--stem", default="generalization_rank_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_figure(args.main_csv, args.shuffle_csv, SIGN_SPECS, args.out_dir, args.stem)


if __name__ == "__main__":
    main()
