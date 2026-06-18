#!/usr/bin/env python3
"""Animate where reservoir neurons sit on tanh over time."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Load a signed connectome, run a simple tanh reservoir, and animate neuron "
            "pre-activations on the tanh activation curve."
        )
    )
    p.add_argument(
        "--connectome",
        default="Connectome/ce_adj_new.npy",
        help="Path to a square connectome/adjacency .npy file.",
    )
    p.add_argument(
        "--inhibition-percent",
        type=float,
        default=None,
        help=(
            "Optional sign-flip percent, matching the sign_test_og_cel experiment: "
            "randomly make this fraction of nonzero edges negative and the rest positive. "
            "Accepts either 20 or 0.20 for 20%%. If omitted, existing connectome signs are untouched."
        ),
    )
    p.add_argument("--target-sr", type=float, default=1.0, help="Target spectral radius for W.")
    p.add_argument("--leak", type=float, default=0.8, help="Leaky update rate.")
    p.add_argument("--input-scale", type=float, default=1.0, help="Scale of random input weights.")
    p.add_argument("--frames", type=int, default=240, help="Number of simulated/animated time steps.")
    p.add_argument("--fps", type=int, default=24, help="Output video frames per second.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--out",
        default="activation_function_neurons.mp4",
        help="Output path. Use .mp4 for ffmpeg video or .gif for a GIF.",
    )
    p.add_argument("--dpi", type=int, default=150, help="Animation DPI.")
    p.add_argument(
        "--xlim",
        type=float,
        default=0.0,
        help="Symmetric x-axis limit. Defaults to an automatic robust limit.",
    )
    p.add_argument(
        "--input-mode",
        choices=("random", "uniform", "sine"),
        default="random",
        help="Input drive over time. random/uniform matches experiments: u[t] ~ Uniform(-1, 1).",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=40.0,
        help="Scatter point size for neuron positions.",
    )
    p.add_argument(
        "--show-inhibitory",
        action="store_true",
        help="Color inhibitory/excitatory neurons separately instead of black points.",
    )
    return p.parse_args()


def normalize_percent(value: float) -> float:
    frac = float(value)
    if frac > 1.0:
        frac /= 100.0
    if not (0.0 <= frac <= 1.0):
        raise ValueError("--inhibition-percent must be between 0 and 1, or between 0 and 100.")
    return frac


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def load_square_matrix(path: Path) -> np.ndarray:
    matrix = np.load(path).astype(np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def spectral_radius(matrix: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def scale_to_spectral_radius(matrix: np.ndarray, target_sr: float) -> tuple[np.ndarray, float, float]:
    sr_before = spectral_radius(matrix)
    if sr_before <= 1e-12:
        if abs(target_sr) <= 1e-12:
            return matrix.copy(), sr_before, 0.0
        raise ValueError("Cannot rescale a zero-spectral-radius matrix to a nonzero target.")
    scaled = matrix * (target_sr / sr_before)
    return scaled, sr_before, spectral_radius(scaled)


def apply_percent_negative_edges(
    matrix: np.ndarray,
    inhibition_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Match util.util.apply_percent_negative for weighted sign-test reservoirs."""
    signed = np.abs(matrix).astype(np.float64, copy=True)
    nz = np.nonzero(signed)
    n_edges = len(nz[0])
    n_negative = int(inhibition_fraction * n_edges)
    if n_negative <= 0:
        return signed

    idx = np.arange(n_edges)
    rng.shuffle(idx)
    sel = (nz[0][idx[:n_negative]], nz[1][idx[:n_negative]])
    signed[sel] *= -1.0
    return signed


def infer_inhibitory_sources(matrix: np.ndarray) -> np.ndarray:
    """Infer source neurons with at least one negative outgoing edge in source-row matrices."""
    return np.any(matrix < 0.0, axis=1)


def make_input(frames: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode in {"random", "uniform"}:
        return rng.uniform(-1.0, 1.0, size=frames)
    t = np.linspace(0.0, 6.0 * np.pi, frames, endpoint=False)
    return 0.85 * np.sin(t) + 0.15 * np.sin(0.37 * t + 0.8)


def simulate(
    W: np.ndarray,
    leak: float,
    input_scale: float,
    frames: int,
    input_mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = W.shape[0]
    win = rng.normal(0.0, input_scale, size=n)
    u = make_input(frames, input_mode, rng)
    z = np.zeros(n, dtype=np.float64)
    pre_history = np.empty((frames, n), dtype=np.float64)
    act_history = np.empty((frames, n), dtype=np.float64)

    for t in range(frames):
        pre = W.T @ z + win * u[t]
        activated = np.tanh(pre)
        z = (1.0 - leak) * z + leak * activated
        pre_history[t] = pre
        act_history[t] = activated

    return pre_history, act_history, u


def robust_xlim(pre_history: np.ndarray) -> float:
    finite = np.abs(pre_history[np.isfinite(pre_history)])
    if finite.size == 0:
        return 3.0
    val = float(np.percentile(finite, 99.0))
    return max(2.5, min(8.0, 1.15 * val))


def save_animation(
    pre_history: np.ndarray,
    act_history: np.ndarray,
    u: np.ndarray,
    inhibitory: np.ndarray,
    out_path: Path,
    title: str,
    fps: int,
    dpi: int,
    xlim: float,
    show_inhibitory: bool,
    point_size: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(-xlim, xlim, 700)
    ys = np.tanh(xs)

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(xs, ys, color="#f12012", linewidth=3.0, solid_capstyle="round", zorder=1)
    ax.axhline(0.0, color="#f12012", linewidth=2.0, alpha=0.75, zorder=0)
    ax.axvline(0.0, color="#f12012", linewidth=2.0, alpha=0.75, zorder=0)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("pre-activation")
    ax.set_ylabel("tanh(pre-activation)")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)

    if show_inhibitory:
        colors = np.where(inhibitory, "#1f77b4", "#111111")
        scat = ax.scatter(
            pre_history[0],
            act_history[0],
            c=colors,
            s=point_size,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
    else:
        scat = ax.scatter(
            pre_history[0],
            act_history[0],
            c="#111111",
            s=point_size,
            alpha=0.92,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )

    def update(frame: int):
        scat.set_offsets(np.column_stack([pre_history[frame], act_history[frame]]))
        saturation = 100.0 * float(np.mean(np.abs(pre_history[frame]) > 2.0))
        linear = 100.0 * float(np.mean(np.abs(pre_history[frame]) <= 0.5))
        time_text.set_text(
            f"t = {frame}\n"
            f"input = {u[frame]:+.3f}\n"
            f"|pre| <= 0.5: {linear:.1f}%\n"
            f"|pre| > 2: {saturation:.1f}%"
        )
        return scat, time_text

    anim = FuncAnimation(fig, update, frames=pre_history.shape[0], interval=1000 / fps, blit=False)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    inhibition_fraction = normalize_percent(args.inhibition_percent) if args.inhibition_percent is not None else None
    connectome_path = Path(args.connectome)
    if not connectome_path.is_file():
        raise FileNotFoundError(f"Connectome file not found: {connectome_path}")

    base = load_square_matrix(connectome_path)
    if inhibition_fraction is None:
        signed = base
        sign_mode = "existing connectome signs preserved"
    else:
        signed = apply_percent_negative_edges(base, inhibition_fraction, rng)
        sign_mode = f"sign_test_og_cel-style random negative edges ({format_percent(inhibition_fraction)})"
    inhibitory = infer_inhibitory_sources(signed)
    W, sr_before, sr_after = scale_to_spectral_radius(signed, args.target_sr)
    pre_history, act_history, u = simulate(W, args.leak, args.input_scale, args.frames, args.input_mode, rng)
    xlim = float(args.xlim) if args.xlim > 0 else robust_xlim(pre_history)

    source_negative_fraction = float(np.mean(inhibitory)) if inhibitory.size else 0.0
    nonzero_edges = signed != 0.0
    negative_edge_fraction = (
        float(np.count_nonzero(signed[nonzero_edges] < 0.0) / np.count_nonzero(nonzero_edges))
        if np.count_nonzero(nonzero_edges)
        else 0.0
    )
    title = (
        f"{connectome_path.name}: neurons on tanh over time\n"
        f"negative edges={format_percent(negative_edge_fraction)}, SR {sr_before:.3g}->{sr_after:.3g}"
    )
    save_animation(
        pre_history,
        act_history,
        u,
        inhibitory,
        Path(args.out),
        title,
        args.fps,
        args.dpi,
        xlim,
        args.show_inhibitory,
        args.point_size,
    )
    print(f"[saved] {args.out}")
    print(f"[info] connectome={connectome_path}")
    print(f"[info] sign mode: {sign_mode}")
    print(
        f"[info] source rows with negative outgoing edge={int(inhibitory.sum())}/{len(inhibitory)} "
        f"({format_percent(source_negative_fraction)})"
    )
    print(f"[info] negative nonzero edges={format_percent(negative_edge_fraction)}")
    print(f"[info] spectral radius: {sr_before:.8g} -> {sr_after:.8g}")
    print("[info] input: all neurons receive random Win_i * u[t], u[t] ~ Uniform(-1, 1)")
    print(f"[info] xlim=+/-{xlim:.4g}")


if __name__ == "__main__":
    main()
