#!/usr/bin/env python3
"""Create a conceptual figure for kernel rank and generalization rank."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "kernel_generalization_rank_concept.png"
PDF_PATH = OUT_DIR / "kernel_generalization_rank_concept.pdf"


def effective_rank(matrix: np.ndarray) -> float:
    """Shannon effective rank of a centered time x neuron matrix."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    singular_values = np.clip(singular_values, 1e-12, None)
    probabilities = singular_values / singular_values.sum()
    entropy = -(probabilities * np.log(probabilities)).sum()
    return float(np.exp(entropy))


def state_matrix(time: np.ndarray, modes: list[tuple[float, float]]) -> np.ndarray:
    """Build a neuron activity matrix from shared temporal modes."""
    columns = []
    for neuron, (freq, phase) in enumerate(modes):
        carrier = np.sin(freq * time + phase)
        slow = 0.35 * np.cos((0.55 + 0.08 * neuron) * time - 0.3 * phase)
        columns.append(carrier + slow)
    return np.column_stack(columns)


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    subtitle: str,
    cmap: str,
    title_size: int = 13,
    subtitle_size: int = 9,
) -> None:
    ax.imshow(matrix.T, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=title_size, fontweight="bold", loc="left", pad=12)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=subtitle_size, color="#4b5563")
    ax.set_xlabel("time")
    ax.set_ylabel("neurons")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_spectrum(ax: plt.Axes, matrix: np.ndarray, color: str) -> None:
    singular_values = np.linalg.svd(matrix - matrix.mean(axis=0, keepdims=True), compute_uv=False)
    normalized = singular_values / singular_values.max()
    ax.bar(np.arange(1, len(normalized) + 1), normalized, color=color, width=0.75)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("state direction")
    ax.set_ylabel("relative variance")
    ax.set_xticks([1, len(normalized)])
    ax.set_yticks([0, 1])
    ax.spines[["top", "right"]].set_visible(False)


def draw_input_panel(ax: plt.Axes, time: np.ndarray) -> None:
    clean = np.sin(1.8 * time) + 0.25 * np.sin(4.1 * time + 0.4)
    noisy = clean + 0.23 * np.sin(14.0 * time + 0.5) + 0.07 * np.cos(22.0 * time)
    ax.plot(time, clean, color="#0f766e", lw=2.0, label="clean input")
    ax.plot(time, noisy, color="#f97316", lw=1.2, alpha=0.9, label="input + noise")
    ax.set_title("Generalization rank", fontsize=13, fontweight="bold", loc="left", pad=8)
    ax.text(
        0.0,
        1.01,
        "Noise is added to the same input signal.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    ax.set_xlabel("time")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(frameon=False, loc="lower left", fontsize=8, ncols=2)
    ax.spines[["top", "right", "left"]].set_visible(False)


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    bbox = ax.get_position()
    fig.text(
        bbox.x0 - 0.035,
        bbox.y1 + 0.01,
        label,
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )


def main() -> None:
    rng = np.random.default_rng(8)
    time = np.linspace(0, 10, 170)

    low_modes = [(1.05, 0.0), (1.05, 0.2), (1.08, 0.4), (1.10, 0.7), (1.02, 1.0), (1.06, 1.2)]
    high_modes = [(0.8, 0.0), (1.3, 1.1), (1.9, 0.4), (2.6, 2.0), (3.4, 1.7), (4.2, 0.9)]
    low_state = state_matrix(time, low_modes)
    high_state = state_matrix(time, high_modes)

    # Add tiny deterministic variation so duplicated modes do not look perfectly copied.
    low_state += 0.03 * rng.normal(size=low_state.shape)
    high_state += 0.03 * rng.normal(size=high_state.shape)

    clean_state = high_state
    common_noise = 0.18 * np.sin(11.0 * time)[:, None] * np.linspace(0.6, 1.1, clean_state.shape[1])
    robust_delta = common_noise + 0.02 * rng.normal(size=clean_state.shape)
    fragile_delta = 0.11 * state_matrix(
        time,
        [(6.5, 0.2), (7.1, 1.4), (8.0, 2.1), (8.8, 0.9), (9.5, 1.9), (10.4, 0.1)],
    )
    fragile_delta += 0.04 * rng.normal(size=clean_state.shape)

    robust_gr = effective_rank(robust_delta)
    fragile_gr = effective_rank(fragile_delta)
    low_kr = effective_rank(low_state)
    high_kr = effective_rank(high_state)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#374151",
            "axes.titlesize": 13,
            "axes.labelsize": 9,
            "xtick.color": "#6b7280",
            "ytick.color": "#6b7280",
        }
    )

    fig = plt.figure(figsize=(12.4, 5.4), constrained_layout=False)
    fig.patch.set_facecolor("#f8fafc")
    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.05, 0.82],
        width_ratios=[1, 1, 1, 1],
        hspace=0.7,
        wspace=0.42,
    )

    fig.suptitle(
        "Kernel rank and generalization rank",
        fontsize=17,
        fontweight="bold",
        x=0.05,
        y=0.98,
        ha="left",
        color="#111827",
    )

    ax_low = fig.add_subplot(gs[0, 0])
    ax_high = fig.add_subplot(gs[0, 1])
    ax_low_spec = fig.add_subplot(gs[1, 0])
    ax_high_spec = fig.add_subplot(gs[1, 1])

    draw_heatmap(ax_low, low_state, "Low kernel rank", f"KR = {low_kr:.1f}", "YlGnBu")
    draw_heatmap(ax_high, high_state, "High kernel rank", f"KR = {high_kr:.1f}", "YlGnBu")
    draw_spectrum(ax_low_spec, low_state, "#0891b2")
    draw_spectrum(ax_high_spec, high_state, "#0f766e")
    ax_low_spec.set_title("few active directions", fontsize=9, color="#4b5563")
    ax_high_spec.set_title("many active directions", fontsize=9, color="#4b5563")

    ax_input = fig.add_subplot(gs[0, 2:])
    draw_input_panel(ax_input, time)

    ax_robust = fig.add_subplot(gs[1, 2])
    ax_fragile = fig.add_subplot(gs[1, 3])
    draw_heatmap(
        ax_robust,
        robust_delta,
        "Robust",
        f"GR = {robust_gr:.1f}",
        "Oranges",
        title_size=11,
        subtitle_size=8,
    )
    draw_heatmap(
        ax_fragile,
        fragile_delta,
        "Noise-sensitive",
        f"GR = {fragile_gr:.1f}",
        "Oranges",
        title_size=11,
        subtitle_size=8,
    )

    add_panel_label(fig, ax_low, "A")
    add_panel_label(fig, ax_input, "B")

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
