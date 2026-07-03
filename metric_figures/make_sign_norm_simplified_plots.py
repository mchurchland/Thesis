"""Create simplified alternatives for the sign-normalization ablation figures.

The existing plots show every metric and every diagnostic.  These alternatives
collapse metrics only after within-dataset normalization, so the figures can
communicate the same normalization mechanism with fewer visual channels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "metric_figures" / "sign_norm_simplified"

DATASETS = {
    "matched_cel": {
        "label": "C. elegans 4:1",
        "path": ROOT / "good_results" / "good_cel_new" / "matched_cel",
        "marker_frac": 0.22058823529411764,
        "marker_label": "observed",
    },
    "matched_er": {
        "label": "ER matched",
        "path": ROOT / "good_results" / "good_cel_new" / "matched_er",
        "marker_frac": 0.2,
        "marker_label": "4:1",
    },
    "removed_cel": {
        "label": "Removed connectome",
        "path": ROOT / "good_results" / "good_cel_new" / "removed_cel",
        "marker_frac": 0.2425287356321839,
        "marker_label": "observed",
    },
}

PRIMARY_DENSE = {
    "label": "C. elegans 4:1",
    "path": ROOT / "good_results" / "good_cel_new" / "matched_cel",
    "marker_frac": 0.22058823529411764,
    "marker_label": r"$q_{\mathrm{orig}}$",
}

DENSE_OPTION8_DATASETS = {
    "cel_4to1": {
        "label": "C. elegans 4:1",
        "path": ROOT / "good_results" / "good_cel_new" / "matched_cel",
        "marker_frac": 0.22058823529411764,
        "marker_label": r"$q_{\mathrm{orig}}$",
        "stem": "option_8_average_perf_cv",
    },
    "er_matched": {
        "label": "ER matched 4:1",
        "path": ROOT / "good_results" / "good_cel_new" / "matched_er",
        "marker_frac": 0.2,
        "marker_label": r"$q_{\mathrm{4:1}}$",
        "stem": "option_8_average_perf_cv_er_matched",
    },
    "removed": {
        "label": "Removed connectome",
        "path": ROOT / "good_results" / "good_cel_new" / "removed_cel",
        "marker_frac": 0.2425287356321839,
        "marker_label": r"$q_{\mathrm{orig}}$",
        "stem": "option_8_average_perf_cv_removed",
    },
}

METRICS = ("MC", "IPC", "KR", "GR")
GROUPS = {
    "All metrics": METRICS,
    "Memory": ("MC", "IPC"),
    "Kernel": ("KR", "GR"),
}
NORM_LABELS = {
    "spectral_radius": "own radius",
    "original_radius": "orig. radius",
}
NORM_COLORS = {
    "spectral_radius": "#276fbf",
    "original_radius": "#c8501a",
}
GROUP_COLORS = {
    "All metrics": "#222222",
    "Memory": "#276fbf",
    "Kernel": "#7a4fb3",
}
METRIC_COLORS = {
    "MC": "#2364AA",
    "IPC": "#E07A1F",
    "KR": "#208A4B",
    "GR": "#7B52AB",
}
NORM_STYLES = {
    "spectral_radius": "-",
    "original_radius": "--",
}


def _load_dataset(info: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(info["path"])
    perf = pd.read_csv(path / "sign_norm_ablation_mean_performance.csv")
    cv = pd.read_csv(path / "sign_norm_ablation_cv.csv")
    scaling = pd.read_csv(path / "sign_norm_ablation_scaling.csv")
    for df in (perf, cv, scaling):
        if "sign_frac" in df:
            df["sign_frac"] = pd.to_numeric(df["sign_frac"], errors="coerce")
    return perf, cv, scaling


def _normalize_metric_rows(df: pd.DataFrame, *, invert: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["index"] = np.nan
    out["index_sem"] = np.nan
    for metric, sub in out.groupby("metric", sort=False):
        lo = float(sub["mean"].min())
        hi = float(sub["mean"].max())
        span = hi - lo
        if span <= 1e-12:
            norm = np.full(len(sub), 0.5)
            sem = np.zeros(len(sub))
        else:
            norm = (sub["mean"].to_numpy(float) - lo) / span
            sem = sub.get("sem", pd.Series(0.0, index=sub.index)).to_numpy(float) / span
        if invert:
            norm = 1.0 - norm
        out.loc[sub.index, "index"] = norm
        out.loc[sub.index, "index_sem"] = sem
    return out


def _group_index(
    df: pd.DataFrame,
    metric_group: tuple[str, ...],
    *,
    invert: bool = False,
) -> pd.DataFrame:
    normed = _normalize_metric_rows(df[df["metric"].isin(metric_group)], invert=invert)
    grouped = (
        normed.groupby(["normalization", "sign_frac"], as_index=False)
        .agg(index=("index", "mean"), index_sem=("index_sem", "mean"))
        .sort_values(["normalization", "sign_frac"])
    )
    return grouped


def _raw_rho_relative(scaling: pd.DataFrame) -> pd.DataFrame:
    raw = (
        scaling[scaling["normalization"] == "spectral_radius"]
        .loc[:, ["sign_frac", "raw_rho"]]
        .dropna()
        .sort_values("sign_frac")
        .copy()
    )
    denom = float(raw["raw_rho"].max()) if not raw.empty else np.nan
    raw["raw_rho_relative"] = raw["raw_rho"] / denom if denom > 0 else np.nan
    return raw


def _rho_ratio(scaling: pd.DataFrame) -> pd.DataFrame:
    base = (
        scaling[scaling["normalization"] == "spectral_radius"]
        .loc[:, ["sign_frac", "raw_rho", "ref_rho"]]
        .dropna()
        .sort_values("sign_frac")
        .copy()
    )
    base["rho_ratio"] = base["raw_rho"] / base["ref_rho"].replace(0.0, np.nan)
    return base


def _crossings(xs: np.ndarray, ys: np.ndarray, level: float = 1.0) -> list[float]:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float) - float(level)
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]
    out: list[float] = []
    for i in range(len(xs) - 1):
        y0 = ys[i]
        y1 = ys[i + 1]
        if abs(y0) <= 1e-12:
            out.append(float(xs[i]))
        if y0 * y1 < 0:
            frac = abs(y0) / (abs(y0) + abs(y1))
            out.append(float(xs[i] + frac * (xs[i + 1] - xs[i])))
    if len(xs) and abs(ys[-1]) <= 1e-12:
        out.append(float(xs[-1]))
    return out


def _draw_marker(ax: plt.Axes, info: dict[str, object], y: float = 0.98) -> None:
    frac = info.get("marker_frac")
    if frac is None:
        return
    frac = float(frac)
    label = str(info.get("marker_label", "observed"))
    ax.axvspan(frac - 0.012, frac + 0.012, color="#f0c94a", alpha=0.20, linewidth=0)
    ax.axvline(frac, color="#7a5a00", linewidth=1.6, alpha=0.75)
    if y >= 0.5:
        xytext = (min(frac + 0.12, 0.92), 0.91)
        va = "center"
    else:
        xytext = (min(frac + 0.12, 0.92), 0.12)
        va = "center"
    ax.annotate(
        label,
        xy=(frac, y),
        xycoords=("data", "axes fraction"),
        xytext=xytext,
        textcoords=("data", "axes fraction"),
        ha="center",
        va=va,
        fontsize=8,
        color="#4d3a00",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#b89200", "alpha": 0.88},
        arrowprops={"arrowstyle": "-|>", "color": "#7a5a00", "lw": 1.0, "shrinkA": 2, "shrinkB": 2},
    )


def _draw_marker_line(ax: plt.Axes, info: dict[str, object]) -> None:
    frac = info.get("marker_frac")
    if frac is None:
        return
    frac = float(frac)
    ax.axvspan(frac - 0.012, frac + 0.012, color="#f0c94a", alpha=0.20, linewidth=0)
    ax.axvline(frac, color="#7a5a00", linewidth=1.6, alpha=0.75)


def _setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _setup_paper_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.grid(True, alpha=0.18, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color("#333333")
    ax.tick_params(axis="both", labelsize=10.5, width=1.0, length=4.2, color="#333333")


def _radius_regions(info: dict[str, object], ratio: pd.DataFrame) -> tuple[float, float]:
    xs = ratio["sign_frac"].to_numpy(float)
    ys = ratio["rho_ratio"].to_numpy(float)
    hits = _crossings(xs, ys, level=1.0)
    if len(hits) >= 2:
        return hits[0], hits[-1]
    q_orig = float(info["marker_frac"])
    return q_orig, 1.0 - q_orig


def _shade_radius_regions(ax: plt.Axes, left: float, right: float) -> None:
    ax.axvspan(-0.02, left, color="#E76F51", alpha=0.055, linewidth=0, zorder=0)
    ax.axvspan(left, right, color="#2A9D8F", alpha=0.075, linewidth=0, zorder=0)
    ax.axvspan(right, 1.02, color="#E76F51", alpha=0.055, linewidth=0, zorder=0)


def _draw_q_reference_lines(ax: plt.Axes, info: dict[str, object]) -> None:
    q_orig = float(info["marker_frac"])
    q_label = str(info.get("marker_label", r"$q_{\mathrm{orig}}$"))
    if q_label.startswith("$") and q_label.endswith("$"):
        opposite_label = "$1-" + q_label[1:]
    else:
        opposite_label = f"1 - {q_label}"
    for xpos, label, ha in (
        (q_orig, q_label, "left"),
        (1.0 - q_orig, opposite_label, "right"),
    ):
        ax.axvline(xpos, color="#A77C00", linewidth=1.25, linestyle="--", alpha=0.72)
        ax.text(
            xpos,
            0.97,
            label,
            transform=ax.get_xaxis_transform(),
            ha=ha,
            va="top",
            fontsize=10.5,
            color="#6C5200",
        )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _normalized_all_metric_index(df: pd.DataFrame) -> pd.DataFrame:
    return _group_index(df, METRICS)


def _draw_overlay_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    scaling: pd.DataFrame,
    *,
    include_marker_label: bool,
    info: dict[str, object],
) -> None:
    index = _group_index(summary, GROUPS["All metrics"])
    for norm in ("spectral_radius", "original_radius"):
        sub = index[index["normalization"] == norm]
        ax.plot(
            sub["sign_frac"],
            sub["index"],
            color=NORM_COLORS[norm],
            linestyle=NORM_STYLES[norm],
            marker="o",
            markersize=4,
            linewidth=2.0,
            label=NORM_LABELS[norm],
        )
    raw = _raw_rho_relative(scaling)
    ax.plot(
        raw["sign_frac"],
        raw["raw_rho_relative"],
        color="#666666",
        linestyle=":",
        marker=".",
        linewidth=2.0,
        label="raw radius",
    )
    if include_marker_label:
        _draw_marker(ax, info)
    else:
        _draw_marker_line(ax, info)
    _setup_axis(ax)


def save_mechanism_overlay(datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), dpi=300, sharex=True, sharey=True)
    for col, (key, (perf, cv, scaling)) in enumerate(datasets.items()):
        info = DATASETS[key]
        _draw_overlay_panel(axes[0, col], perf, scaling, include_marker_label=True, info=info)
        _draw_overlay_panel(axes[1, col], cv, scaling, include_marker_label=False, info=info)
        axes[0, col].set_title(str(info["label"]), fontsize=11)
        axes[1, col].set_xlabel("Negative edge fraction")
    axes[0, 0].set_ylabel("Metric-normalized performance / raw radius")
    axes[1, 0].set_ylabel("Metric-normalized CV / raw radius")
    handles = [
        Line2D([0], [0], color=NORM_COLORS["spectral_radius"], linestyle="-", marker="o", label="Own-radius normalization"),
        Line2D([0], [0], color=NORM_COLORS["original_radius"], linestyle="--", marker="o", label="Original-radius normalization"),
        Line2D([0], [0], color="#666666", linestyle=":", marker=".", label="Raw radius, relative"),
        Line2D([0], [0], color="#7a5a00", linewidth=1.6, label="Observed / target fraction"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Option 1: performance and CV mechanism overlay", y=1.06, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "option_1_mechanism_overlay.png", bbox_inches="tight")
    plt.close(fig)


def _delta_rows(df: pd.DataFrame, metric_group: tuple[str, ...]) -> pd.DataFrame:
    sub = df[df["metric"].isin(metric_group)].copy()
    wide = sub.pivot_table(
        index=["sign_frac", "metric"],
        columns="normalization",
        values="mean",
        aggfunc="mean",
    ).reset_index()
    wide = wide.dropna(subset=["spectral_radius", "original_radius"])
    wide["delta_pct"] = (
        (wide["spectral_radius"] - wide["original_radius"])
        / wide["original_radius"].abs().clip(lower=1e-12)
        * 100.0
    )
    return (
        wide.groupby("sign_frac", as_index=False)["delta_pct"]
        .mean()
        .sort_values("sign_frac")
    )


def save_normalization_delta(datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=300, sharey=False)
    for ax, (key, (perf, _cv, _scaling)) in zip(axes, datasets.items(), strict=True):
        info = DATASETS[key]
        for group_name in ("All metrics", "Memory", "Kernel"):
            sub = _delta_rows(perf, GROUPS[group_name])
            ax.plot(
                sub["sign_frac"],
                sub["delta_pct"],
                color=GROUP_COLORS[group_name],
                linestyle="-" if group_name == "All metrics" else "--",
                marker="o",
                markersize=4,
                linewidth=2.0,
                label=group_name,
            )
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        _draw_marker(ax, info, y=0.04)
        _setup_axis(ax)
        ax.set_title(str(info["label"]), fontsize=11)
        ax.set_xlabel("Negative edge fraction")
    axes[0].set_ylabel("Performance change: own radius vs orig. radius (%)")
    handles = [
        Line2D([0], [0], color=GROUP_COLORS["All metrics"], marker="o", linewidth=2.0, label="All metrics"),
        Line2D([0], [0], color=GROUP_COLORS["Memory"], marker="o", linestyle="--", linewidth=2.0, label="Memory"),
        Line2D([0], [0], color=GROUP_COLORS["Kernel"], marker="o", linestyle="--", linewidth=2.0, label="Kernel"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Option 2: plot only the normalization effect", y=1.12, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "option_2_normalization_delta.png", bbox_inches="tight")
    plt.close(fig)


def save_grouped_perf_cv(datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.1), dpi=300, sharex=True)
    for col, (key, (perf, cv, _scaling)) in enumerate(datasets.items()):
        info = DATASETS[key]
        for row, (source_df, invert, ylabel) in enumerate(
            (
                (perf, False, "Metric-normalized performance"),
                (cv, False, "Metric-normalized CV (higher = more sensitive)"),
            )
        ):
            ax = axes[row, col]
            for group_name in ("Memory", "Kernel"):
                group = _group_index(source_df, GROUPS[group_name], invert=invert)
                for norm in ("spectral_radius", "original_radius"):
                    sub = group[group["normalization"] == norm]
                    ax.plot(
                        sub["sign_frac"],
                        sub["index"],
                        color=GROUP_COLORS[group_name],
                        linestyle=NORM_STYLES[norm],
                        marker="o",
                        markersize=3.8,
                        linewidth=1.8,
                    )
            _draw_marker(ax, info)
            _setup_axis(ax)
            if row == 0:
                ax.set_title(str(info["label"]), fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Negative edge fraction")
    handles = [
        Line2D([0], [0], color=GROUP_COLORS["Memory"], linewidth=2.0, label="Memory"),
        Line2D([0], [0], color=GROUP_COLORS["Kernel"], linewidth=2.0, label="Kernel"),
        Line2D([0], [0], color="#333333", linestyle="-", marker="o", linewidth=1.8, label="own radius"),
        Line2D([0], [0], color="#333333", linestyle="--", marker="o", linewidth=1.8, label="orig. radius"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Option 3: grouped performance and CV", y=1.06, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "option_3_grouped_perf_cv.png", bbox_inches="tight")
    plt.close(fig)


def save_radius_mechanism_only(primary: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    _perf, _cv, scaling = primary
    info = PRIMARY_DENSE
    ratio = _rho_ratio(scaling)
    left, right = _radius_regions(info, ratio)

    fig, ax = plt.subplots(figsize=(4.8, 3.1), dpi=300)
    _shade_radius_regions(ax, left, right)
    ax.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.85)
    ax.plot(
        ratio["sign_frac"],
        ratio["rho_ratio"],
        color="#4B5563",
        marker="o",
        markersize=3.7,
        linewidth=2.0,
        zorder=3,
    )
    _draw_q_reference_lines(ax, info)
    ax.text(
        0.50,
        0.10,
        r"$\rho(W)<\rho(W_{\mathrm{orig}})$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#1F6F64",
    )
    ax.text(
        1.0,
        1.0,
        "target",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#222222",
    )
    ax.set_xlabel("Negative edge fraction, $q$", fontsize=9.5)
    ax.set_ylabel(r"$\rho(W) / \rho(W_{\mathrm{orig}})$", fontsize=9.5)
    #ax.set_ylim(0.58, 1.56)
    _setup_paper_axis(ax)
    fig.tight_layout()
    _save_figure(fig, "option_4_radius_mechanism_only")


def _draw_norm_index_lines(ax: plt.Axes, indexed: pd.DataFrame) -> None:
    for norm in ("spectral_radius", "original_radius"):
        sub = indexed[indexed["normalization"] == norm].sort_values("sign_frac")
        if sub.empty:
            continue
        ax.plot(
            sub["sign_frac"],
            sub["index"],
            color=NORM_COLORS[norm],
            linestyle=NORM_STYLES[norm],
            marker="o",
            markersize=3.4,
            linewidth=1.8,
        )


def _label_norm_endpoints(ax: plt.Axes, indexed: pd.DataFrame) -> None:
    labels = {
        "spectral_radius": r"scale by $\rho(W)$",
        "original_radius": r"scale by $\rho(W_{\mathrm{orig}})$",
    }
    for norm in ("spectral_radius", "original_radius"):
        sub = indexed[indexed["normalization"] == norm].sort_values("sign_frac")
        if sub.empty:
            continue
        last = sub.iloc[-1]
        ax.text(
            1.018,
            float(last["index"]),
            labels[norm],
            ha="left",
            va="center",
            fontsize=8.0,
            color=NORM_COLORS[norm],
            clip_on=False,
        )


def _draw_grouped_performance_panel(
    ax: plt.Axes,
    perf: pd.DataFrame,
    group_name: str,
    metrics: tuple[str, ...],
) -> None:
    for metric in metrics:
        for norm in ("spectral_radius", "original_radius"):
            sub = perf[(perf["metric"] == metric) & (perf["normalization"] == norm)].sort_values("sign_frac")
            if sub.empty:
                continue
            ax.plot(
                sub["sign_frac"],
                sub["mean"],
                color=METRIC_COLORS[metric],
                linestyle=NORM_STYLES[norm],
                marker="o",
                markersize=3.2,
                linewidth=1.75,
                alpha=0.92,
            )
    for metric in metrics:
        metric_rows = perf[perf["metric"] == metric].sort_values("sign_frac")
        if metric_rows.empty:
            continue
        label_y = float(metric_rows[metric_rows["normalization"] == "spectral_radius"]["mean"].iloc[-1])
        ax.text(
            1.012,
            label_y,
            metric,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.2,
            color=METRIC_COLORS[metric],
            fontweight="semibold",
            clip_on=False,
        )
    ax.text(
        0.02,
        0.92,
        group_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#222222",
        fontweight="semibold",
    )
    ax.set_ylabel("Performance", fontsize=9.0)
    _setup_paper_axis(ax)


def save_compact_mechanism_response(primary: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    perf, _cv, scaling = primary
    info = PRIMARY_DENSE
    ratio = _rho_ratio(scaling)
    left, right = _radius_regions(info, ratio)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(5.25, 6.35),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.08, 1.0, 1.0], "hspace": 0.16},
    )

    ax = axes[0]
    _shade_radius_regions(ax, left, right)
    ax.axhline(1.0, color="#222222", linewidth=0.95, linestyle="--", alpha=0.85)
    ax.plot(ratio["sign_frac"], ratio["rho_ratio"], color="#4B5563", marker="o", markersize=3.3, linewidth=1.9)
    _draw_q_reference_lines(ax, info)
    ax.set_title("C. elegans 4:1 sign-normalization response", loc="left", fontsize=9.6, pad=4)
    ax.set_ylabel(r"$\rho(W) / \rho(W_{\mathrm{orig}})$", fontsize=9)
    #ax.set_ylim(0.58, 1.56)
    _setup_paper_axis(ax)

    for ax, group_name in ((axes[1], "Memory"), (axes[2], "Kernel")):
        _shade_radius_regions(ax, left, right)
        _draw_grouped_performance_panel(ax, perf, group_name, GROUPS[group_name])
        ax.set_ylim(bottom=0.0)
    axes[1].set_xlabel("")
    axes[2].set_xlabel("Negative edge fraction, $q$", fontsize=9.5)
    axes[0].text(
        0.985,
        0.92,
        "radius ratio",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#4B5563",
    )
    norm_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=1.75, label=r"scale by $\rho(W)$"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            linewidth=1.75,
            label=r"scale by $\rho(W_{\mathrm{orig}})$",
        ),
    ]
    axes[1].legend(
        handles=norm_handles,
        loc="lower left",
        frameon=False,
        fontsize=7.0,
        handlelength=2.6,
        borderaxespad=0.2,
    )
    fig.tight_layout()
    _save_figure(fig, "option_5_compact_mechanism_response")


def save_metric_delta_panels(primary: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    perf, _cv, scaling = primary
    info = PRIMARY_DENSE
    ratio = _rho_ratio(scaling)
    left, right = _radius_regions(info, ratio)

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.8), dpi=300, sharex=True)
    axes_flat = axes.ravel()
    for idx, metric in enumerate(METRICS):
        ax = axes_flat[idx]
        sub = _delta_rows(perf, (metric,))
        _shade_radius_regions(ax, left, right)
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.75)
        q_orig = float(info["marker_frac"])
        ax.axvline(q_orig, color="#A77C00", linewidth=0.9, linestyle="--", alpha=0.58)
        ax.axvline(1.0 - q_orig, color="#A77C00", linewidth=0.9, linestyle="--", alpha=0.58)
        ax.plot(
            sub["sign_frac"],
            sub["delta_pct"],
            color=METRIC_COLORS[metric],
            marker="o",
            markersize=3.3,
            linewidth=1.8,
        )
        ax.set_title(metric, fontsize=10.2, pad=2)
        _setup_paper_axis(ax)
        ax.margins(y=0.08)
        if idx in (0, 2):
            ax.set_ylabel("Own - original (%)", fontsize=8.8)
        if idx in (2, 3):
            ax.set_xlabel("Negative edge fraction, $q$", fontsize=9.2)

    handles = [
        Patch(facecolor="#2A9D8F", alpha=0.14, edgecolor="none", label=r"$\rho(W)<\rho(W_{\mathrm{orig}})$"),
        Line2D([0], [0], color="#A77C00", linestyle="--", linewidth=1.0, label=r"$q_{\mathrm{orig}},\,1-q_{\mathrm{orig}}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 1.02), fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, "option_6_metric_delta_panels")


def save_mechanism_plus_effect(primary: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    perf, _cv, scaling = primary
    info = PRIMARY_DENSE
    ratio = _rho_ratio(scaling)
    left, right = _radius_regions(info, ratio)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), dpi=300)

    ax = axes[0]
    _shade_radius_regions(ax, left, right)
    ax.axhline(1.0, color="#222222", linewidth=0.95, linestyle="--", alpha=0.85)
    ax.plot(ratio["sign_frac"], ratio["rho_ratio"], color="#4B5563", marker="o", markersize=3.5, linewidth=1.9)
    _draw_q_reference_lines(ax, info)
    ax.set_xlabel("Negative edge fraction, $q$", fontsize=9.2)
    ax.set_ylabel(r"$\rho(W) / \rho(W_{\mathrm{orig}})$", fontsize=9.2)
    #ax.set_ylim(0.58, 1.56)
    _setup_paper_axis(ax)

    ax = axes[1]
    _shade_radius_regions(ax, left, right)
    ax.axhline(0.0, color="#333333", linewidth=0.85, alpha=0.78)
    for group_name in ("All metrics", "Memory", "Kernel"):
        sub = _delta_rows(perf, GROUPS[group_name])
        ax.plot(
            sub["sign_frac"],
            sub["delta_pct"],
            color=GROUP_COLORS[group_name],
            linestyle="-" if group_name == "All metrics" else "--",
            marker="o",
            markersize=3.3,
            linewidth=1.8,
            label=group_name,
        )
    q_orig = float(info["marker_frac"])
    ax.axvline(q_orig, color="#A77C00", linewidth=0.9, linestyle="--", alpha=0.58)
    ax.axvline(1.0 - q_orig, color="#A77C00", linewidth=0.9, linestyle="--", alpha=0.58)
    ax.set_xlabel("Negative edge fraction, $q$", fontsize=9.2)
    ax.set_ylabel("Own - original performance (%)", fontsize=9.2)
    _setup_paper_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.20),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=2.3,
        columnspacing=1.0,
    )

    fig.tight_layout(w_pad=1.6, rect=[0, 0, 1, 0.93])
    _save_figure(fig, "option_7_mechanism_plus_effect")


def _draw_group_average_index_panel(
    ax: plt.Axes,
    source_df: pd.DataFrame,
    *,
    panel_label: str,
    ylabel: str,
) -> None:
    for group_name in ("Memory", "Kernel"):
        grouped = _group_index(source_df, GROUPS[group_name])
        for norm in ("spectral_radius", "original_radius"):
            sub = grouped[grouped["normalization"] == norm].sort_values("sign_frac")
            if sub.empty:
                continue
            ax.plot(
                sub["sign_frac"],
                sub["index"],
                color=GROUP_COLORS[group_name],
                linestyle=NORM_STYLES[norm],
                marker="o",
                markersize=2.8,
                linewidth=1.55,
                alpha=0.94,
            )
        own = grouped[grouped["normalization"] == "spectral_radius"].sort_values("sign_frac")
        if not own.empty:
            last = own.iloc[-1]
            ax.text(
                1.012,
                float(last["index"]),
                group_name,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.4,
                color=GROUP_COLORS[group_name],
                clip_on=False,
            )
    ax.set_title(panel_label, loc="left", fontsize=8.6, pad=2)
    ax.set_ylabel(ylabel, fontsize=8.4)
    ax.set_ylim(-0.04, 1.04)
    _setup_paper_axis(ax)


def _draw_group_average_raw_panel(
    ax: plt.Axes,
    source_df: pd.DataFrame,
    *,
    panel_label: str,
    ylabel: str,
) -> None:
    endpoint_labels: list[tuple[str, float]] = []
    for group_name in ("Memory", "Kernel"):
        grouped = (
            source_df[source_df["metric"].isin(GROUPS[group_name])]
            .groupby(["normalization", "sign_frac"], as_index=False)
            .agg(mean=("mean", "mean"))
            .sort_values(["normalization", "sign_frac"])
        )
        for norm in ("spectral_radius", "original_radius"):
            sub = grouped[grouped["normalization"] == norm]
            if sub.empty:
                continue
            ax.plot(
                sub["sign_frac"],
                sub["mean"],
                color=GROUP_COLORS[group_name],
                linestyle=NORM_STYLES[norm],
                marker="o",
                markersize=4.5,
                linewidth=2.35,
                alpha=0.94,
            )
        own = grouped[grouped["normalization"] == "spectral_radius"].sort_values("sign_frac")
        if not own.empty:
            last = own.iloc[-1]
            endpoint_labels.append((group_name, float(last["mean"])))
    ax.set_title(panel_label, loc="left", fontsize=12.0, fontweight="semibold", pad=3)
    ax.set_ylabel(ylabel, fontsize=11.4)
    ax.margins(y=0.08)
    _setup_paper_axis(ax)
    if endpoint_labels:
        y0, y1 = ax.get_ylim()
        yrange = max(y1 - y0, 1e-12)
        min_sep = 0.070 * yrange
        adjusted = sorted(endpoint_labels, key=lambda item: item[1])
        if len(adjusted) == 2 and adjusted[1][1] - adjusted[0][1] < min_sep:
            center = 0.5 * (adjusted[0][1] + adjusted[1][1])
            lo = max(y0 + 0.035 * yrange, center - 0.5 * min_sep)
            hi = min(y1 - 0.035 * yrange, center + 0.5 * min_sep)
            adjusted = [(adjusted[0][0], lo), (adjusted[1][0], hi)]
        for group_name, y_pos in adjusted:
            ax.text(
                1.012,
                y_pos,
                group_name,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=10.5,
                color=GROUP_COLORS[group_name],
                clip_on=False,
            )


def save_option_8_average_perf_cv(
    primary: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    info: dict[str, object] | None = None,
    stem: str = "option_8_average_perf_cv",
) -> None:
    perf, cv, scaling = primary
    info = info or PRIMARY_DENSE
    ratio = _rho_ratio(scaling)
    left, right = _radius_regions(info, ratio)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.35, 7.0),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.24},
    )

    ax = axes[0]
    _shade_radius_regions(ax, left, right)
    ax.axhline(1.0, color="#222222", linewidth=1.35, linestyle="--", alpha=0.85)
    ax.plot(
        ratio["sign_frac"],
        ratio["rho_ratio"],
        color="#4B5563",
        marker="o",
        markersize=4.6,
        linewidth=2.45,
    )
    _draw_q_reference_lines(ax, info)
    ax.set_title(
        f"{info['label']} sign-normalization sweep",
        loc="left",
        fontsize=13.2,
        fontweight="semibold",
        pad=4,
    )
    ax.set_ylabel(r"$\rho(W) / \rho(W_{\mathrm{orig}})$", fontsize=11.4)
    #ax.set_ylim(0.58, 1.56)
    _setup_paper_axis(ax)

    _shade_radius_regions(axes[1], left, right)
    _draw_group_average_raw_panel(axes[1], perf, panel_label="Mean performance", ylabel="Performance")

    _shade_radius_regions(axes[2], left, right)
    _draw_group_average_raw_panel(axes[2], cv, panel_label="Mean coefficient of variation", ylabel="CV")

    axes[1].set_xlabel("")
    axes[2].set_xlabel("Negative edge fraction, $q$", fontsize=12.5, labelpad=7)

    norm_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.4, label=r"scale by $\rho(W)$"),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            linewidth=2.4,
            label=r"scale by $\rho(W_{\mathrm{orig}})$",
        ),
    ]
    fig.legend(
        handles=norm_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.020),
        ncol=2,
        frameon=False,
        fontsize=10.2,
        handlelength=2.8,
        columnspacing=1.6,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.890,
        bottom=0.165,
        top=0.955,
        hspace=0.285,
    )
    _save_figure(fig, stem)


def save_companion_summary(datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> None:
    rows = []
    for key, (perf, cv, scaling) in datasets.items():
        for group_name, metrics in GROUPS.items():
            perf_delta = _delta_rows(perf, metrics)
            cv_delta = _delta_rows(cv, metrics)
            for _, row in perf_delta.iterrows():
                rows.append(
                    {
                        "dataset": key,
                        "summary": "performance_delta_pct",
                        "metric_group": group_name,
                        "sign_frac": row["sign_frac"],
                        "value": row["delta_pct"],
                    }
                )
            for _, row in cv_delta.iterrows():
                rows.append(
                    {
                        "dataset": key,
                        "summary": "cv_delta_pct",
                        "metric_group": group_name,
                        "sign_frac": row["sign_frac"],
                        "value": row["delta_pct"],
                    }
                )
        raw = _raw_rho_relative(scaling)
        for _, row in raw.iterrows():
            rows.append(
                {
                    "dataset": key,
                    "summary": "raw_rho_relative",
                    "metric_group": "diagnostic",
                    "sign_frac": row["sign_frac"],
                    "value": row["raw_rho_relative"],
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "simplified_summary.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {key: _load_dataset(info) for key, info in DATASETS.items()}
    primary_dense = _load_dataset(PRIMARY_DENSE)
    dense_option8 = {
        key: _load_dataset(info)
        for key, info in DENSE_OPTION8_DATASETS.items()
    }
    save_mechanism_overlay(datasets)
    save_normalization_delta(datasets)
    save_grouped_perf_cv(datasets)
    save_radius_mechanism_only(primary_dense)
    save_compact_mechanism_response(primary_dense)
    save_metric_delta_panels(primary_dense)
    save_mechanism_plus_effect(primary_dense)
    for key, info in DENSE_OPTION8_DATASETS.items():
        save_option_8_average_perf_cv(
            dense_option8[key],
            info=info,
            stem=str(info["stem"]),
        )
    save_companion_summary(datasets)
    print(f"saved simplified sign-normalization plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
