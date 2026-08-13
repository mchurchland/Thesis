#!/usr/bin/env python3
"""Quick look at sign-sweep behavior for C. elegans and ER-matched networks."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

if not os.environ.get("MPLBACKEND"):
    mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_raw_rho_performance_summary import build_summary


OUT_DIR = Path("good_results/summary")
OUT_STEM = "sign_sweep_quicklook"
CE_ONLY_OUT_STEM = "sign_sweep_quicklook_celegans_only"
REF_SIGN_FRAC = 0.2425287356321839


def _aggregate_sign_sweeps(summary: pd.DataFrame) -> pd.DataFrame:
    sign = summary[summary["series_kind"] == "sign_sweep"].copy()
    sign = sign[sign["dataset"].isin(["Matched C. elegans sweep", "Matched ER sweep"])].copy()
    sign["sign_frac"] = pd.to_numeric(sign["sign_frac"], errors="coerce")
    sign = sign[np.isfinite(sign["sign_frac"])].copy()

    agg = (
        sign.groupby(["dataset", "sign_frac"], as_index=False)
        .agg(
            raw_rho=("raw_rho", "mean"),
            performance=("performance", "mean"),
            cv=("cv", "mean"),
        )
        .sort_values(["dataset", "sign_frac"])
        .reset_index(drop=True)
    )

    return agg


def _load_ce_reference_rho(summary: pd.DataFrame) -> float:
    ce_mask = (
        (summary["series_kind"] == "shuffle")
        & (summary["mode"] == "real")
    )
    if not ce_mask.any():
        raise ValueError("Could not locate the original C. elegans rho in the summary data.")
    return float(summary.loc[ce_mask, "raw_rho"].iloc[0])


def _load_reference_rhos(summary: pd.DataFrame) -> tuple[float, float]:
    ce_original_rho = _load_ce_reference_rho(summary)

    er_mask = (
        (summary["series_kind"] == "sign_sweep")
        & (summary["dataset"] == "Matched ER sweep")
    )
    if not er_mask.any():
        raise ValueError("Could not locate the ER rho at the C. elegans sign fraction.")
    er_points = (
        summary.loc[er_mask, ["sign_frac", "raw_rho"]]
        .dropna()
        .astype(float)
        .sort_values("sign_frac")
    )
    er_at_ce_rho = float(np.interp(REF_SIGN_FRAC, er_points["sign_frac"], er_points["raw_rho"]))

    return ce_original_rho, er_at_ce_rho


def _save_celegans_only(agg: pd.DataFrame, ce_original_rho: float) -> None:
    ce_curve = (
        agg[agg["dataset"] == "Matched C. elegans sweep"]
        .sort_values("sign_frac")
        .copy()
    )
    if ce_curve.empty:
        raise ValueError("No matched C. elegans sign-sweep rows found.")

    color = "#d55e00"
    y_min = min(float(ce_curve["raw_rho"].min()), ce_original_rho)
    y_max = max(float(ce_curve["raw_rho"].max()), ce_original_rho)
    y_pad = max(3.0, (y_max - y_min) * 0.08)

    fig, ax = plt.subplots(figsize=(7.35, 3.65), dpi=300)
    ax.plot(
        ce_curve["sign_frac"],
        ce_curve["raw_rho"],
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.2,
        zorder=3,
    )
    ax.axhline(
        ce_original_rho,
        color=color,
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        zorder=2,
    )
    ax.axvline(
        REF_SIGN_FRAC,
        color="#444444",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=2,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("E/I Edge Balance")
    ax.set_ylabel(r"raw $\rho(W)$")
    ax.set_title("Mean raw spectral radius", fontweight="semibold", pad=5)
    ax.grid(True, axis="y", color="#dfdfdf", linewidth=0.65, alpha=0.7)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.9)

    ax.text(
        1.005,
        ce_original_rho,
        "C. elegans original",
        color=color,
        fontsize=9.4,
        va="center",
        ha="left",
        transform=ax.get_yaxis_transform(),
    )
    ax.text(
        REF_SIGN_FRAC + 0.012,
        y_max + y_pad * 0.72,
        "empirical E/I Edge Balance",
        fontsize=9.4,
        color="#444444",
        va="top",
        ha="left",
    )
    ax.text(
        0.70,
        0.90,
        "C. elegans sweep",
        color=color,
        fontsize=10.6,
        fontweight="semibold",
        transform=ax.transAxes,
        ha="left",
        va="center",
    )

    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.105, right=0.92)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{CE_ONLY_OUT_STEM}.png"
    pdf = OUT_DIR / f"{CE_ONLY_OUT_STEM}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def main(*, celegans_only: bool = False) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.9,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    summary = build_summary(
        "good_results/shuf/combined.ALL.csv",
        "good_results/good_cel_new",
        max_sign_frac=1.0,
    )
    agg = _aggregate_sign_sweeps(summary)
    if agg.empty:
        raise ValueError("No sign-sweep rows found in the summary data.")
    if celegans_only:
        ce_original_rho = _load_ce_reference_rho(summary)
        _save_celegans_only(agg, ce_original_rho)
        return

    ce_original_rho, er_at_ce_rho = _load_reference_rhos(summary)
    raw_max = float(agg["raw_rho"].max())
    er_max = float(agg.loc[agg["dataset"] == "Matched ER sweep", "raw_rho"].max())
    lower_max = max(20.0, er_max * 1.45)
    upper_min = max(35.0, ce_original_rho * 0.34)

    colors = {
        "Matched C. elegans sweep": "#d55e00",
        "Matched ER sweep": "#56b4e9",
    }

    fig, (ax_hi, ax_lo) = plt.subplots(
        2,
        1,
        figsize=(7.35, 4.95),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    for ax in (ax_hi, ax_lo):
        for dataset, sub in agg.groupby("dataset", sort=False):
            sub = sub.sort_values("sign_frac")
            color = colors.get(dataset, "#333333")
            ax.plot(
                sub["sign_frac"],
                sub["raw_rho"],
                color=color,
                linewidth=2.4,
                marker="o",
                markersize=6.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.2,
                label=dataset,
                zorder=3,
            )
        ax.axvline(REF_SIGN_FRAC, color="#444444", linestyle="--", linewidth=1.2, alpha=0.9, zorder=2)
        ax.grid(True, axis="y", color="#dfdfdf", linewidth=0.65, alpha=0.7)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.9)

    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(labelbottom=False, bottom=False)
    ax_lo.tick_params(top=False)

    ax_hi.set_ylim(upper_min, raw_max * 1.06)
    ax_lo.set_ylim(0.0, lower_max)
    ax_lo.set_xlim(-0.02, 1.02)
    ax_lo.set_xlabel("E/I Edge Balance")
    ax_lo.set_ylabel(r"raw $\rho(W)$")
    ax_hi.set_ylabel(r"raw $\rho(W)$")
    ax_hi.set_title("Mean raw spectral radius", fontweight="semibold", pad=5)

    ax_hi.axhline(ce_original_rho, color="#d55e00", linestyle="--", linewidth=1.8, alpha=0.95, zorder=2)
    ax_lo.axhline(er_at_ce_rho, color="#56b4e9", linestyle=":", linewidth=1.9, alpha=0.95, zorder=2)

    d = 0.008
    kwargs_hi = dict(transform=ax_hi.transAxes, color="k", clip_on=False, linewidth=1.0)
    kwargs_lo = dict(transform=ax_lo.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_hi.plot((-d, +d), (-d, +d), **kwargs_hi)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kwargs_hi)
    ax_lo.plot((-d, +d), (1 - d, 1 + d), **kwargs_lo)
    ax_lo.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_lo)

    ax_hi.text(1.005, ce_original_rho, "C. elegans original", color="#d55e00", fontsize=9.4, va="center", ha="left", transform=ax_hi.get_yaxis_transform())
    ax_lo.text(1.005, er_at_ce_rho, "ER at CE balance", color="#56b4e9", fontsize=9.4, va="center", ha="left", transform=ax_lo.get_yaxis_transform())
    ax_lo.text(
        REF_SIGN_FRAC + 0.012,
        lower_max * 0.94,
        "empirical E/I Edge Balance",
        fontsize=9.4,
        color="#444444",
        va="top",
        ha="left",
    )

    c_curve = agg[(agg["dataset"] == "Matched C. elegans sweep")].sort_values("sign_frac")
    drop_point = c_curve.loc[c_curve["sign_frac"].astype(float).sub(REF_SIGN_FRAC).abs().idxmin()]
    ax_lo.annotate(
        "drop before\nempirical balance",
        xy=(float(drop_point["sign_frac"]), float(drop_point["raw_rho"])),
        xytext=(0.34, min(14.0, lower_max * 0.8)),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#d55e00", lw=1.3),
        fontsize=9.2,
        color="#d55e00",
        ha="left",
        va="center",
    )

    ax_hi.text(
        0.70,
        0.92,
        "C. elegans sweep",
        color="#d55e00",
        fontsize=10.6,
        fontweight="semibold",
        transform=ax_hi.transAxes,
        ha="left",
        va="center",
    )
    ax_lo.text(
        0.70,
        0.74,
        "ER sweep",
        color="#56b4e9",
        fontsize=10.6,
        fontweight="semibold",
        transform=ax_lo.transAxes,
        ha="left",
        va="center",
    )
    fig.subplots_adjust(top=0.925, bottom=0.125, left=0.095, right=0.945, hspace=0.08)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{OUT_STEM}.png"
    pdf = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


if __name__ == "__main__":
    main()
