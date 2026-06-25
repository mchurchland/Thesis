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
    save_mechanism_overlay(datasets)
    save_normalization_delta(datasets)
    save_grouped_perf_cv(datasets)
    save_companion_summary(datasets)
    print(f"saved simplified sign-normalization plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
