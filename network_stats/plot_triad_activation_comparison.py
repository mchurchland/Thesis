#!/usr/bin/env python3
"""Plot 0% versus 50% sign-fraction reservoir activation counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    ("active_state_count_mean", "Total active neuron-time samples"),
    ("activation_onset_count_mean", "Activation onsets"),
)
COLORS = {0.0: "#0072B2", 0.5: "#D55E00"}


def _comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"sign_frac", "raw_rho_mean", *(column for column, _ in METRICS)}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Activation summary is missing columns: {missing}")

    df = summary.copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["sign_frac"].isin((0.0, 0.5))].dropna(subset=list(required))
    if "normalization" in df.columns:
        spectral = df[df["normalization"].astype(str) == "spectral_radius"]
        if not spectral.empty:
            df = spectral
    if df.empty:
        raise ValueError("No finite 0% and 50% activation rows were found.")
    return df


def plot_comparison(summary_csv: Path, out_path: Path) -> None:
    df = _comparison_table(pd.read_csv(summary_csv))
    sign_fracs = [frac for frac in (0.0, 0.5) if np.isclose(df["sign_frac"], frac).any()]
    if len(sign_fracs) != 2:
        raise ValueError("Both sign fractions (0.0 and 0.5) are required for this comparison.")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), dpi=300)
    offsets = np.linspace(-0.11, 0.11, max(df.get("rho_target", pd.Series()).nunique(), 1))
    for ax, (column, ylabel) in zip(axes, METRICS):
        means, errors, raw_rhos = [], [], []
        for frac in sign_fracs:
            sub = df[np.isclose(df["sign_frac"], frac)].copy()
            values = sub[column].to_numpy(float)
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            raw_rhos.append(float(np.mean(sub["raw_rho_mean"])))

        xs = np.arange(len(sign_fracs))
        ax.bar(
            xs,
            means,
            yerr=errors,
            width=0.62,
            capsize=5,
            color=[COLORS[frac] for frac in sign_fracs],
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        for pos, frac in enumerate(sign_fracs):
            sub = df[np.isclose(df["sign_frac"], frac)].copy()
            values = sub[column].to_numpy(float)
            jitter = np.resize(offsets, len(values))
            ax.scatter(
                np.full(len(values), pos, dtype=float) + jitter,
                values,
                s=32,
                color="white",
                edgecolor=COLORS[frac],
                linewidth=1.2,
                zorder=3,
            )

        ax.set_xticks(xs, [f"{int(frac * 100)}% negative signs\nraw $\\rho$ = {rho:.1f}" for frac, rho in zip(sign_fracs, raw_rhos)])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("A  Sustained activity", loc="left", fontweight="bold")
    axes[1].set_title("B  Separate activation events", loc="left", fontweight="bold")
    fig.suptitle("Higher raw spectral radius is associated with fewer reservoir activations", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Bars: mean across spectral-radius targets; error bars: SD across targets; points: individual targets.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        default="network_stats/triad_sign_fraction_results/triad_sign_fraction_group_summary.ALL.csv",
    )
    parser.add_argument(
        "--out",
        default="network_stats/triad_sign_fraction_results/triad_activation_comparison_0_vs_50.png",
    )
    args = parser.parse_args()
    plot_comparison(Path(args.summary_csv), Path(args.out))


if __name__ == "__main__":
    main()
