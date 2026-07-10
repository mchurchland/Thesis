#!/usr/bin/env python3
"""Plot normalized weight histograms for two CE sign-balance settings."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import torch

from util.util import (
    assign_random_unknown_signs,
    build_reservoir,
    load_connectome,
    load_unknown_sign_weights,
)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    feature_conn: str
    ce_adj: str
    out_dir: str
    title: str
    use_unknown_4to1: bool = False
    ce_unknown_sign_weights: str | None = None


DATASETS = {
    "new_4to1": DatasetSpec(
        label="new_4to1",
        feature_conn="sign_test_og_cel",
        ce_adj="Connectome/ce_adj_unk41.npy",
        ce_unknown_sign_weights="Connectome/ce_unknown_sign_weights_unk41.npy",
        out_dir="good_results/signtest_final_new/new_4to1",
        title="4:1 unknown-random",
        use_unknown_4to1=True,
    ),
    "new_removed": DatasetSpec(
        label="new_removed",
        feature_conn="sign_test_og_cel",
        ce_adj="Connectome/ce_adj_removed.npy",
        out_dir="good_results/signtest_final_new/new_removed",
        title="Removed-edge",
    ),
    "matched_er_4to1": DatasetSpec(
        label="matched_er_4to1",
        feature_conn="sign_test_er",
        ce_adj="Connectome/ce_adj_unk41.npy",
        ce_unknown_sign_weights="Connectome/ce_unknown_sign_weights_unk41.npy",
        out_dir="good_results/signtest_final_new/er_matched",
        title="Matched ER, 4:1 unknown-random",
        use_unknown_4to1=True,
    ),
}


def _normalized_sign_balance_weights(
    ce_W_trial: np.ndarray,
    *,
    feature_conn: str,
    sign_frac: float,
    seed: int,
    rho_target: float,
    normalization: str,
) -> tuple[np.ndarray, dict[str, float | str]]:
    nnz_target = int(np.count_nonzero(ce_W_trial)) if feature_conn == "sign_test_er" else None
    Wt, _, info = build_reservoir(
        feature_conn=feature_conn,
        target_sr=float(rho_target),
        N=ce_W_trial.shape[0],
        ce_W_bio=ce_W_trial,
        input_scale=1.0,
        seed=int(seed),
        nnz_target=nnz_target,
        DEVICE=torch.device("cpu"),
        per_neg=float(sign_frac),
        normalization_mode=normalization,
        normalization_ref=ce_W_trial,
        return_info=True,
    )
    values = Wt.detach().cpu().numpy().reshape(-1)
    return values[values != 0], info


def _load_trial_connectome(args: argparse.Namespace, spec: DatasetSpec) -> np.ndarray:
    ce_W_base, _, _ = load_connectome(spec.ce_adj, None)
    if ce_W_base is None:
        raise FileNotFoundError(f"Could not load CE adjacency: {spec.ce_adj}")
    if not spec.use_unknown_4to1:
        return ce_W_base

    unknown_weights = load_unknown_sign_weights(
        spec.ce_adj,
        spec.ce_unknown_sign_weights,
        n_nodes=ce_W_base.shape[0],
    )
    if unknown_weights is None:
        raise FileNotFoundError(
            "Could not load unknown-sign weights for the 4:1 completion. "
            "Pass --ce-unknown-sign-weights."
        )

    ce_W_trial = assign_random_unknown_signs(
        ce_W_base,
        unknown_weights,
        np.random.default_rng(args.seed + args.unknown_sign_seed_offset),
        inhibitory_fraction=args.unknown_sign_inhibitory_frac,
    )
    return ce_W_trial


def _log_edges(min_abs: float, max_abs: float, n_bins: int) -> np.ndarray:
    return np.logspace(np.log10(min_abs * 0.85), np.log10(max_abs), max(6, int(n_bins)) + 1)


def _weight_ticks(min_abs: float, max_abs: float) -> list[float]:
    candidates = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    lo = min_abs * 0.85
    hi = max_abs
    return [float(v) for v in candidates if lo <= v <= hi]


def _nonzero_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, keep in enumerate(mask):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _plot_nonzero_stairs(
    ax: plt.Axes,
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    color: str,
    label: str,
) -> None:
    first = True
    for start, end in _nonzero_runs(counts > 0):
        run_counts = counts[start:end]
        ax.step(
            edges[start : end + 1],
            np.r_[run_counts, run_counts[-1]],
            where="post",
            linewidth=1.35,
            color=color,
            label=label if first else "_nolegend_",
        )
        first = False


def plot_histograms(args: argparse.Namespace, spec: DatasetSpec) -> str:
    ce_W_trial = _load_trial_connectome(args, spec)
    conditions = [
        (0.0, "0% negative"),
        (0.50, "50% negative"),
    ]
    weights = []
    infos = []
    for sign_frac, label in conditions:
        vals, info = _normalized_sign_balance_weights(
            ce_W_trial,
            feature_conn=spec.feature_conn,
            sign_frac=sign_frac,
            seed=args.seed + 1,
            rho_target=args.rho_target,
            normalization=args.normalization,
        )
        weights.append((vals, label))
        infos.append(info)

    abs_weights = [(np.abs(vals[np.abs(vals) > 0]), label) for vals, label in weights]
    all_abs_vals = np.concatenate([vals for vals, _ in abs_weights])
    max_abs = max(1.0, float(np.max(all_abs_vals)))
    min_abs = float(np.min(all_abs_vals[all_abs_vals > 0]))
    edges = _log_edges(min_abs, max_abs, args.bins)
    ticks = _weight_ticks(min_abs, max_abs)
    histogram_counts = []
    for vals, _ in abs_weights:
        counts, _ = np.histogram(vals, bins=edges)
        histogram_counts.append(counts)
    positive_counts = np.concatenate([counts[counts > 0] for counts in histogram_counts])
    y_min = max(10.0, float(np.percentile(positive_counts, 5)) * 0.8)
    y_max = float(np.max(positive_counts)) * 1.18

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.15, 2.65), dpi=300)
        colors = ["#0072B2", "#D55E00"]
        for counts, (_, label), color in zip(histogram_counts, abs_weights, colors):
            _plot_nonzero_stairs(ax, counts, edges, color=color, label=label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min_abs * 0.85, max_abs)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:g}" for tick in ticks])
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis="y", which="major", color="0.88", linewidth=0.55)
        ax.tick_params(axis="both", which="major", length=3.2, pad=1.5)
        ax.tick_params(axis="both", which="minor", length=1.8)
        ax.set_xlabel("Absolute normalized edge weight", labelpad=1.5)
        ax.set_ylabel("Edge count", labelpad=1.5)
        normalization_label = args.normalization.replace("_", "-")
        ax.set_title(
            rf"{spec.title}; {normalization_label} normalized ($\rho={args.rho_target:g}$)",
            pad=3,
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.83),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="0.82",
            framealpha=0.96,
            handlelength=1.5,
            borderpad=0.28,
            labelspacing=0.25,
            columnspacing=0.9,
        )
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.18, top=0.64)

    out_dir = args.out_dir or spec.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    fig.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    for (_, label), info in zip(weights, infos):
        print(
            f"{spec.label} | {label}: raw_rho={float(info['raw_rho']):.6g}, "
            f"post_rho={float(info['post_rho']):.6g}, "
            f"scale_factor={float(info['scale_factor']):.6g}"
        )
    print(f"[saved] {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare normalized nonzero edge-weight histograms for empirical "
            "C. elegans sign balance and 50% sign balance."
        )
    )
    p.add_argument(
        "--dataset",
        choices=tuple(DATASETS) + ("all",),
        default="new_4to1",
        help="Preset model/dataset to plot. Use 'all' to generate all presets.",
    )
    p.add_argument("--unknown-sign-inhibitory-frac", type=float, default=0.2)
    p.add_argument("--unknown-sign-seed-offset", type=int, default=23_000_000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--rho-target", type=float, default=1.0)
    p.add_argument("--normalization", choices=("spectral_radius", "original_radius"), default="spectral_radius")
    p.add_argument("--bins", type=int, default=8, help="Number of logarithmic bins.")
    p.add_argument("--out-dir", default="", help="Override output directory. Not valid with --dataset all.")
    p.add_argument("--out-name", default="sign_balance_weight_histograms.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "all":
        if args.out_dir:
            raise ValueError("--out-dir override is ambiguous with --dataset all.")
        for spec in DATASETS.values():
            plot_histograms(args, spec)
        return
    plot_histograms(args, DATASETS[args.dataset])


if __name__ == "__main__":
    main()
