#!/usr/bin/env python3
"""Create a conceptual figure for task-agnostic reservoir metrics."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "task_agnostic_metrics_concept.png"
PDF_PATH = OUT_DIR / "task_agnostic_metrics_concept.pdf"


def add_card(ax: plt.Axes, title: str, subtitle: str, color: str) -> None:
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    card = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#e5e7eb",
        linewidth=1.2,
        zorder=-10,
    )
    ax.add_patch(card)
    ax.text(0.06, 0.9, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color=color)
    ax.text(0.06, 0.79, subtitle, transform=ax.transAxes, fontsize=9, color="#4b5563")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_schematic(ax: plt.Axes, time: np.ndarray) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Same reservoir, no task labels", loc="left", fontsize=14, fontweight="bold", pad=10)

    x = np.linspace(0.05, 0.33, len(time))
    u = 0.52 + 0.11 * np.sin(1.7 * time) + 0.05 * np.sin(4.8 * time + 0.4)
    ax.plot(x, u, color="#0f766e", lw=2.4)
    ax.text(0.06, 0.72, "input", fontsize=10, color="#0f766e", fontweight="bold")

    arrow = FancyArrowPatch((0.36, 0.54), (0.48, 0.54), arrowstyle="-|>", mutation_scale=18, lw=1.8, color="#64748b")
    ax.add_patch(arrow)

    nodes = {
        0: (0.56, 0.62),
        1: (0.67, 0.74),
        2: (0.78, 0.62),
        3: (0.77, 0.43),
        4: (0.63, 0.36),
        5: (0.53, 0.46),
    }
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.plot([xa, xb], [ya, yb], color="#94a3b8", lw=1.3, alpha=0.85)
    for i, (xn, yn) in nodes.items():
        color = "#f97316" if i in (1, 3) else "#0891b2"
        ax.add_patch(Circle((xn, yn), 0.032, facecolor=color, edgecolor="white", lw=1.2, zorder=5))
    ax.text(0.58, 0.21, "reservoir states", fontsize=10, color="#334155", fontweight="bold")


def draw_memory(ax: plt.Axes, time: np.ndarray) -> None:
    add_card(ax, "Memory capacity", "past input is recoverable", "#0f766e")
    y = 0.43 + 0.2 * np.sin(1.8 * time)
    ax.plot(np.linspace(0.08, 0.92, len(time)), y, color="#0f766e", lw=2.0)
    for x0 in [0.28, 0.43, 0.58]:
        ax.annotate(
            "",
            xy=(x0 - 0.13, 0.38),
            xytext=(x0, 0.56),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#0f766e", alpha=0.9),
        )
    ax.text(0.09, 0.25, "u(t - tau)", transform=ax.transAxes, fontsize=11, fontweight="bold", color="#0f766e")


def draw_ipc(ax: plt.Axes, time: np.ndarray) -> None:
    add_card(ax, "Information processing", "nonlinear transforms are recoverable", "#7c2d12")
    x = np.linspace(0.08, 0.92, len(time))
    u = np.sin(1.7 * time)
    ax.plot(x, 0.55 + 0.1 * u, color="#f97316", lw=1.7, label="u")
    ax.plot(x, 0.35 + 0.14 * (u**3), color="#9a3412", lw=2.1, label="u^3")
    ax.text(0.09, 0.2, "u, u^2, u^3, ...", transform=ax.transAxes, fontsize=11, fontweight="bold", color="#9a3412")


def draw_kernel_rank(ax: plt.Axes, rng: np.random.Generator) -> None:
    add_card(ax, "Kernel rank", "state diversity over time", "#155e75")
    mat = np.vstack(
        [
            np.sin(np.linspace(0, 6, 80)),
            np.sin(np.linspace(0, 9, 80) + 0.8),
            np.cos(np.linspace(0, 7, 80) - 0.3),
            np.sin(np.linspace(0, 12, 80) + 1.5),
            np.cos(np.linspace(0, 5, 80) + 1.0),
        ]
    )
    mat += 0.05 * rng.normal(size=mat.shape)
    ax.imshow(mat, extent=(0.08, 0.92, 0.17, 0.68), aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax.text(0.09, 0.08, "many state directions", transform=ax.transAxes, fontsize=11, fontweight="bold", color="#155e75")


def draw_gr(ax: plt.Axes, rng: np.random.Generator) -> None:
    add_card(ax, "Generalization rank", "state change under input noise", "#9a3412")
    base = np.sin(np.linspace(0, 16, 90))[None, :]
    weights = np.linspace(0.65, 1.05, 5)[:, None]
    delta = weights * base + 0.08 * rng.normal(size=(5, 90))
    ax.imshow(delta, extent=(0.08, 0.92, 0.17, 0.68), aspect="auto", cmap="Oranges", interpolation="nearest")
    ax.text(0.09, 0.08, "X_noisy - X_clean", transform=ax.transAxes, fontsize=11, fontweight="bold", color="#9a3412")


def main() -> None:
    rng = np.random.default_rng(4)
    time = np.linspace(0, 10, 180)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#374151",
            "xtick.color": "#6b7280",
            "ytick.color": "#6b7280",
        }
    )

    fig = plt.figure(figsize=(12.4, 6.2), constrained_layout=False)
    fig.patch.set_facecolor("#f8fafc")
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[0.95, 1.05], hspace=0.34, wspace=0.28)

    fig.suptitle(
        "Task-agnostic performance metrics",
        fontsize=18,
        fontweight="bold",
        x=0.05,
        y=0.98,
        ha="left",
        color="#111827",
    )
    fig.text(0.05, 0.925, "Probe what the reservoir can represent before choosing a downstream task.", fontsize=10, color="#4b5563")

    schematic = fig.add_subplot(gs[0, :2])
    draw_schematic(schematic, time)

    summary = fig.add_subplot(gs[0, 2:])
    summary.axis("off")
    summary.text(0.0, 0.72, "Train simple readouts from state activity", fontsize=15, fontweight="bold", color="#111827")
    summary.text(0.0, 0.5, "Measure memory, nonlinear processing, diversity, and robustness.", fontsize=12, color="#4b5563")
    summary.text(0.0, 0.23, "No labels. No task-specific objective.", fontsize=17, fontweight="bold", color="#0f172a")

    draw_memory(fig.add_subplot(gs[1, 0]), time)
    draw_ipc(fig.add_subplot(gs[1, 1]), time)
    draw_kernel_rank(fig.add_subplot(gs[1, 2]), rng)
    draw_gr(fig.add_subplot(gs[1, 3]), rng)

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
