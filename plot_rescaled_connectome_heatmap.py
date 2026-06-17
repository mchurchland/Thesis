#!/usr/bin/env python3
"""Plot a spectral-radius-rescaled connectome adjacency matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load a connectome .npy file, rescale it to a target spectral radius, and save a heatmap."
    )
    p.add_argument(
        "--connectome",
        required=True,
        help="Path to a square adjacency/connectome .npy file.",
    )
    p.add_argument(
        "--target-sr",
        type=float,
        default=1.0,
        help="Target spectral radius after rescaling.",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output PNG path. Defaults to '<connectome-stem>_sr<target>_heatmap.png' beside the input.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI.",
    )
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Symmetric color-limit percentile on abs(weight), useful when a few edges are outliers.",
    )
    p.add_argument(
        "--near-zero-thresholds",
        type=float,
        nargs="*",
        default=[0.01, 0.05, 0.1, 0.3],
        help="Absolute rescaled-weight thresholds to annotate in the distribution panel.",
    )
    p.add_argument(
        "--remove-unit-weights",
        action="store_true",
        help="Before spectral-radius rescaling, set entries with abs(weight) == 1 to zero.",
    )
    return p.parse_args()


def spectral_radius(matrix: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def rescale_to_spectral_radius(matrix: np.ndarray, target_sr: float) -> tuple[np.ndarray, float, float]:
    sr_before = spectral_radius(matrix)
    if sr_before <= 1e-12:
        if abs(target_sr) <= 1e-12:
            return matrix.copy(), sr_before, 0.0
        raise ValueError("Cannot rescale a zero-spectral-radius matrix to a nonzero target.")
    scaled = matrix * (float(target_sr) / sr_before)
    sr_after = spectral_radius(scaled)
    return scaled, sr_before, sr_after


def default_output_path(connectome_path: Path, target_sr: float) -> Path:
    sr_label = f"{target_sr:g}".replace(".", "p").replace("-", "m")
    return connectome_path.with_name(f"{connectome_path.stem}_sr{sr_label}_heatmap.png")


def finite_square_matrix(path: Path) -> np.ndarray:
    matrix = np.load(path).astype(np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    return matrix


def remove_unit_weights(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    out = matrix.copy()
    mask = np.isclose(np.abs(out), 1.0)
    removed = int(np.count_nonzero(mask))
    out[mask] = 0.0
    return out, removed


def _log_or_linear_bins(values: np.ndarray, n_bins: int = 60) -> np.ndarray:
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo > 0.0 and hi > lo:
        return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    return np.linspace(lo, hi if hi > lo else lo + 1.0, n_bins + 1)


def _format_distribution_stats(matrix: np.ndarray, abs_nonzero: np.ndarray, thresholds: list[float]) -> str:
    total = int(matrix.size)
    zero_count = int(np.count_nonzero(matrix == 0.0))
    lines = [
        f"exact zeros: {100.0 * zero_count / max(total, 1):.1f}% ({zero_count}/{total})",
        f"nonzero edges: {abs_nonzero.size}",
    ]
    if abs_nonzero.size:
        lines.append(f"median |w|: {np.median(abs_nonzero):.4g}")
        for threshold in thresholds:
            if threshold > 0:
                frac_le = 100.0 * float(np.mean(abs_nonzero <= threshold))
                frac_gt = 100.0 - frac_le
                lines.append(f"|w| <= {threshold:g}: {frac_le:.1f}%")
                lines.append(f"|w| > {threshold:g}: {frac_gt:.1f}%")
    return "\n".join(lines)


def plot_heatmap(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    dpi: int,
    clip_percentile: float,
    near_zero_thresholds: list[float],
) -> str:
    abs_nonzero = np.abs(matrix[np.nonzero(matrix)])
    if abs_nonzero.size:
        pct = float(np.clip(clip_percentile, 0.0, 100.0))
        vmax = float(np.percentile(abs_nonzero, pct))
        if vmax <= 0.0 or not np.isfinite(vmax):
            vmax = float(np.max(abs_nonzero))
    else:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("black")
    shown = np.ma.masked_where(matrix == 0.0, matrix)

    fig, (ax, ax_dist) = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        dpi=dpi,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 0.48], "wspace": 0.28},
    )
    im = ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("source neuron index")
    ax.set_ylabel("target neuron index")
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rescaled edge weight")

    if abs_nonzero.size:
        bins = _log_or_linear_bins(abs_nonzero)
        ax_dist.hist(abs_nonzero, bins=bins, color="#4c78a8", alpha=0.78)
        if bins[0] > 0 and bins[-1] > bins[0]:
            ax_dist.set_xscale("log")
        ax_dist.set_xlabel("|rescaled weight|, nonzero edges")
        ax_dist.set_ylabel("edge count", color="#4c78a8")
        ax_dist.tick_params(axis="y", labelcolor="#4c78a8")
        ax_dist.grid(True, which="both", axis="x", alpha=0.18, linestyle=":")

        sorted_abs = np.sort(abs_nonzero)
        cdf = np.arange(1, sorted_abs.size + 1, dtype=float) / sorted_abs.size
        ax_cdf = ax_dist.twinx()
        ax_cdf.plot(sorted_abs, cdf, color="#e15759", linewidth=2.4)
        ax_cdf.set_ylim(0.0, 1.0)
        ax_cdf.set_ylabel("cumulative fraction", color="#e15759")
        ax_cdf.tick_params(axis="y", labelcolor="#e15759")
        for threshold in near_zero_thresholds:
            if threshold > 0:
                ax_dist.axvline(threshold, color="#333333", linewidth=1.0, linestyle="--", alpha=0.45)
    else:
        ax_dist.text(0.5, 0.5, "no nonzero edges", ha="center", va="center", transform=ax_dist.transAxes)
        ax_dist.set_axis_off()

    ax_dist.set_title("Weight magnitude distribution")
    stats_text = _format_distribution_stats(matrix, abs_nonzero, near_zero_thresholds)
    ax_dist.text(
        0.03,
        0.97,
        stats_text,
        transform=ax_dist.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return stats_text


def main() -> None:
    args = parse_args()
    connectome_path = Path(args.connectome)
    if not connectome_path.is_file():
        raise FileNotFoundError(f"Connectome file not found: {connectome_path}")
    if args.target_sr < 0:
        raise ValueError("--target-sr must be non-negative.")

    matrix = finite_square_matrix(connectome_path)
    removed_unit_weights = 0
    if args.remove_unit_weights:
        matrix, removed_unit_weights = remove_unit_weights(matrix)
    scaled, sr_before, sr_after = rescale_to_spectral_radius(matrix, args.target_sr)
    out_path = Path(args.out) if args.out else default_output_path(connectome_path, args.target_sr)

    title = (
        f"{connectome_path.name} rescaled adjacency\n"
        f"spectral radius: {sr_before:.4g} -> {sr_after:.4g}"
    )
    stats_text = plot_heatmap(scaled, out_path, title, args.dpi, args.clip_percentile, args.near_zero_thresholds)
    print(f"[saved] {out_path}")
    if args.remove_unit_weights:
        print(f"[info] removed abs(weight) == 1 entries before rescaling: {removed_unit_weights}")
    print(f"[info] spectral radius: {sr_before:.8g} -> {sr_after:.8g}")
    print("[info] distribution:")
    for line in stats_text.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    main()
