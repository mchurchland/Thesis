#!/usr/bin/env python3
"""Create replacement figures for the oral presentation slides."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent

BG = "#f8fafc"
INK = "#0f172a"
MUTED = "#475569"
BORDER = "#cbd5e1"
SLATE = "#33464d"
TEAL = "#0f766e"
BLUE = "#2563eb"
ORANGE = "#ea580c"


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "white",
    edgecolor: str = BORDER,
    linewidth: float = 1.6,
    radius: float = 0.035,
    zorder: int = 1,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def pill(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    width: float,
    height: float = 0.105,
    fontsize: int = 22,
) -> None:
    rounded_box(
        ax,
        (x, y),
        width,
        height,
        facecolor="white",
        edgecolor=color,
        linewidth=1.8,
        radius=0.03,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight="bold",
        zorder=5,
    )


def down_arrow(ax: plt.Axes, y0: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (0.5, y0),
            (0.5, y1),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=26,
            linewidth=2.4,
            color="#64748b",
            zorder=2,
        )
    )


def draw_ipc_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0), constrained_layout=False)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Box 1: past input.
    x, y, w, h = 0.08, 0.70, 0.84, 0.20
    rounded_box(ax, (x, y), w, h, facecolor="white", edgecolor="#e2e8f0", linewidth=1.4, radius=0.04)
    ax.plot([x, x], [y + 0.03, y + h - 0.03], transform=ax.transAxes, color=TEAL, lw=5, solid_capstyle="round")
    ax.text(0.16, y + 0.10, "Past\ninput", transform=ax.transAxes, fontsize=18, fontweight="bold", color=TEAL, ha="left", va="center", linespacing=0.9)
    past_labels = [r"$u(t-1)$", r"$u(t-2)$", r"$u(t-3)$"]
    for idx, label in enumerate(past_labels):
        pill(ax, 0.34 + idx * 0.21, y + 0.047, label, color=TEAL, width=0.16, fontsize=22)

    # Box 2: nonlinear target functions.
    x, y, w, h = 0.08, 0.39, 0.84, 0.22
    rounded_box(ax, (x, y), w, h, facecolor="white", edgecolor="#e2e8f0", linewidth=1.4, radius=0.04)
    ax.plot([x, x], [y + 0.03, y + h - 0.03], transform=ax.transAxes, color=ORANGE, lw=5, solid_capstyle="round")
    ax.text(0.16, y + 0.11, "Nonlinear\ntargets", transform=ax.transAxes, fontsize=18, fontweight="bold", color=ORANGE, ha="left", va="center", linespacing=0.9)
    target_specs = [
        (0.30, 0.18, r"$u(t-1)^2$"),
        (0.505, 0.22, r"$u(t-1)u(t-2)$"),
        (0.75, 0.17, r"$P_3(u(t-\tau))$"),
    ]
    for x0, width, label in target_specs:
        pill(ax, x0, y + 0.055, label, color=ORANGE, width=width, fontsize=17)

    # Box 3: the readout question.
    x, y, w, h = 0.08, 0.09, 0.84, 0.20
    rounded_box(ax, (x, y), w, h, facecolor="white", edgecolor="#e2e8f0", linewidth=1.4, radius=0.04)
    ax.plot([x, x], [y + 0.03, y + h - 0.03], transform=ax.transAxes, color=BLUE, lw=5, solid_capstyle="round")
    ax.text(0.16, y + 0.10, "Question", transform=ax.transAxes, fontsize=18, fontweight="bold", color=BLUE, ha="left", va="center")
    ax.text(
        0.31,
        y + 0.10,
        "Can a linear readout reconstruct\nthese from reservoir state?",
        transform=ax.transAxes,
        fontsize=16.5,
        color=INK,
        va="center",
        linespacing=1.12,
    )
    rounded_box(ax, (0.765, y + 0.052), 0.13, 0.095, facecolor="#eff6ff", edgecolor=BLUE, linewidth=1.5, radius=0.025, zorder=4)
    ax.text(
        0.83,
        y + 0.10,
        r"$w^\top x(t)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=19,
        color=INK,
        fontweight="bold",
        zorder=5,
    )

    down_arrow(ax, 0.695, 0.625)
    down_arrow(ax, 0.375, 0.305)

    fig.savefig(OUT_DIR / "ipc_schematic.png", dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(OUT_DIR / "ipc_schematic.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_value_pills(ax: plt.Axes, values: list[str], *, x: float, y: float, width: float, color: str) -> None:
    count = len(values)
    gap = 0.012
    pill_w = (width - gap * (count - 1)) / count
    for idx, value in enumerate(values):
        rounded_box(
            ax,
            (x + idx * (pill_w + gap), y),
            pill_w,
            0.078,
            facecolor="white",
            edgecolor=color,
            linewidth=1.35,
            radius=0.02,
            zorder=4,
        )
        ax.text(
            x + idx * (pill_w + gap) + pill_w / 2,
            y + 0.039,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=INK,
            zorder=5,
        )


def draw_hyperparameter_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.75), constrained_layout=False)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.055, 0.925, "Actual sweep values", transform=ax.transAxes, fontsize=21, fontweight="bold", color=INK)
    ax.text(0.055, 0.865, "Each architecture is evaluated across the same three reservoir-control knobs.", transform=ax.transAxes, fontsize=12.5, color=MUTED)

    cards = [
        {
            "title": "Spectral radius",
            "symbol": r"$\rho(W^{res})$",
            "values": ["0.60", "0.80", "0.95", "1.05"],
            "intuition": "Recurrent gain:\nlarger values keep activity\ncirculating longer; near 1\ntests the edge of stability.",
            "color": TEAL,
        },
        {
            "title": "Leak",
            "symbol": r"$\alpha$",
            "values": ["0.60", "0.80", "1.00"],
            "intuition": "Update speed:\nsmaller leak mixes in the\nnew state more slowly;\nalpha = 1 uses full update.",
            "color": BLUE,
        },
        {
            "title": "Input scaling",
            "symbol": r"$s_{in}$",
            "values": ["0.10", "0.50", "1.00", "1.50"],
            "intuition": "Input drive:\nlarger drive pushes richer\nnonlinear response;\ntoo large can saturate.",
            "color": ORANGE,
        },
    ]

    left = 0.055
    gap = 0.026
    card_w = (0.89 - 2 * gap) / 3
    card_h = 0.58
    y = 0.245

    for idx, card in enumerate(cards):
        x = left + idx * (card_w + gap)
        color = card["color"]
        rounded_box(ax, (x, y), card_w, card_h, facecolor="white", edgecolor="#e2e8f0", linewidth=1.45, radius=0.035)
        ax.plot([x + 0.025, x + card_w - 0.025], [y + card_h - 0.055, y + card_h - 0.055], transform=ax.transAxes, color=color, lw=3.4, solid_capstyle="round")
        ax.text(x + 0.035, y + card_h - 0.095, card["title"], transform=ax.transAxes, fontsize=16, fontweight="bold", color=color)
        ax.text(x + 0.035, y + card_h - 0.155, card["symbol"], transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK)
        ax.text(x + 0.035, y + card_h - 0.235, "values", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#64748b")
        draw_value_pills(ax, card["values"], x=x + 0.035, y=y + card_h - 0.34, width=card_w - 0.07, color=color)
        ax.text(x + 0.035, y + 0.195, "intuition", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color="#64748b")
        ax.text(
            x + 0.035,
            y + 0.155,
            card["intuition"],
            transform=ax.transAxes,
            fontsize=10.8,
            color=INK,
            va="top",
            linespacing=1.18,
        )

    rounded_box(ax, (0.055, 0.08), 0.89, 0.105, facecolor=SLATE, edgecolor=SLATE, linewidth=1.0, radius=0.028)
    ax.text(
        0.5,
        0.132,
        "4 spectral radii x 3 leaks x 4 input scales = 48 settings per architecture / seed",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="white",
        fontweight="bold",
    )

    fig.savefig(OUT_DIR / "hyperparameter_values_intuition.png", dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(OUT_DIR / "hyperparameter_values_intuition.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_triplet(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    values: list[str],
    color: str,
    width: float,
) -> None:
    gap = 0.018
    box_w = (width - 2 * gap) / 3
    for idx, value in enumerate(values):
        rounded_box(
            ax,
            (x + idx * (box_w + gap), y),
            box_w,
            0.12,
            facecolor="white",
            edgecolor=color,
            linewidth=1.9,
            radius=0.028,
            zorder=4,
        )
        ax.text(
            x + idx * (box_w + gap) + box_w / 2,
            y + 0.06,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=21,
            fontweight="bold",
            color=INK,
            zorder=5,
        )


def draw_cv_card(
    ax: plt.Axes,
    *,
    x: float,
    title: str,
    values: list[str],
    mean_text: str,
    cv_text: str,
    label: str,
    label_color: str,
) -> None:
    y = 0.25
    w = 0.38
    h = 0.55
    rounded_box(ax, (x, y), w, h, facecolor="white", edgecolor="#e2e8f0", linewidth=1.45, radius=0.035)
    ax.plot([x + 0.026, x + w - 0.026], [y + h - 0.06, y + h - 0.06], transform=ax.transAxes, color=label_color, lw=3.2, solid_capstyle="round")
    ax.text(x + 0.04, y + h - 0.115, title, transform=ax.transAxes, fontsize=19, fontweight="bold", color=label_color)

    draw_triplet(ax, x=x + 0.04, y=y + h - 0.275, values=values, color=label_color, width=w - 0.08)

    ax.text(x + 0.04, y + 0.215, mean_text, transform=ax.transAxes, fontsize=16, color=INK, fontweight="bold")
    ax.text(x + 0.04, y + 0.15, r"$CV=\sigma/\mu$", transform=ax.transAxes, fontsize=18, color=INK, fontweight="bold")
    ax.text(x + 0.04, y + 0.085, cv_text, transform=ax.transAxes, fontsize=16, color=INK, fontweight="bold")

    rounded_box(ax, (x + w - 0.17, y + 0.065), 0.13, 0.115, facecolor=label_color, edgecolor=label_color, linewidth=1.0, radius=0.026, zorder=5)
    ax.text(
        x + w - 0.105,
        y + 0.122,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
        zorder=6,
    )


def draw_cv_relative_variation_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=False)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.055, 0.91, "Coefficient of variation compares relative spread", transform=ax.transAxes, fontsize=19, fontweight="bold", color=INK)
    ax.text(0.055, 0.845, "Same absolute STD, different scale.", transform=ax.transAxes, fontsize=12.5, color=MUTED)

    draw_cv_card(
        ax,
        x=0.06,
        title="Model A",
        values=["1", "2", "3"],
        mean_text=r"$\mu=2,\ \sigma=1$",
        cv_text=r"$1/2=50\%$",
        label="HIGH",
        label_color=ORANGE,
    )
    draw_cv_card(
        ax,
        x=0.56,
        title="Model B",
        values=["101", "102", "103"],
        mean_text=r"$\mu=102,\ \sigma=1$",
        cv_text=r"$1/102\approx1\%$",
        label="LOW",
        label_color=TEAL,
    )

    ax.add_patch(
        FancyArrowPatch(
            (0.455, 0.53),
            (0.545, 0.53),
            transform=ax.transAxes,
            arrowstyle="<->",
            mutation_scale=18,
            linewidth=1.9,
            color="#64748b",
            zorder=3,
        )
    )
    ax.text(
        0.5,
        0.57,
        r"same $\sigma$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color=MUTED,
        fontweight="bold",
    )

    rounded_box(ax, (0.06, 0.08), 0.88, 0.10, facecolor=SLATE, edgecolor=SLATE, linewidth=1.0, radius=0.026)
    ax.text(
        0.5,
        0.13,
        r"$CV=\sigma/\mu$: same spread can be high or low variation depending on the mean.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12.2,
        color="white",
        fontweight="bold",
    )

    fig.savefig(OUT_DIR / "cv_relative_variation.png", dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(OUT_DIR / "cv_relative_variation.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.labelcolor": "#374151",
            "xtick.color": "#6b7280",
            "ytick.color": "#6b7280",
        }
    )
    draw_ipc_schematic()
    draw_hyperparameter_figure()
    draw_cv_relative_variation_figure()
    print(OUT_DIR / "ipc_schematic.png")
    print(OUT_DIR / "ipc_schematic.pdf")
    print(OUT_DIR / "hyperparameter_values_intuition.png")
    print(OUT_DIR / "hyperparameter_values_intuition.pdf")
    print(OUT_DIR / "cv_relative_variation.png")
    print(OUT_DIR / "cv_relative_variation.pdf")


if __name__ == "__main__":
    main()
