#!/usr/bin/env python3
"""
Plot input/network/Legendre time-series for a single reservoir run.

Panels:
1) input over time
2) average network signal over time
3) Legendre P2 (even) over time
4) Legendre P3 (odd) over time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from inv_arc_test import _pick_device
from network_stats.run_one import run_reservoir_with_pre
from network_stats.stats_util import legendre_P, ridge_fit_predict, r2_score
from util.util import build_reservoir, load_connectome, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot input/network/Legendre traces over time.")
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy")
    p.add_argument("--model", choices=["cel", "er"], default="cel")
    p.add_argument("--er-p", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho", type=float, default=0.95)
    p.add_argument("--leak", type=float, default=0.8)
    p.add_argument("--input-scale", type=float, default=1.0)

    p.add_argument("--washout", type=int, default=12)
    p.add_argument("--t-train", type=int, default=40)
    p.add_argument("--t-test", type=int, default=12)
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument("--cuda", type=int, default=None)
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI for publication-quality export.")
    p.add_argument("--font-base", type=float, default=14.0, help="Base font size.")
    p.add_argument("--font-title", type=float, default=18.0, help="Figure title font size.")
    p.add_argument("--font-small", type=float, default=11.0, help="Small annotation font size.")

    p.add_argument(
        "--out-png",
        default="ipc_order_sweep_ce_many/input_network_legendre_timeseries.png",
        help="Output figure path.",
    )
    p.add_argument(
        "--out-csv",
        default="ipc_order_sweep_ce_many/input_network_legendre_timeseries.csv",
        help="Output data CSV path.",
    )
    return p.parse_args()


def to_m11(u: torch.Tensor) -> torch.Tensor:
    umax = u.max()
    umin = u.min()
    return (2.0 * (u - umin) / (umax - umin + 1e-12)) - 1.0


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": args.font_base,
            "axes.titlesize": args.font_base + 3,
            "axes.labelsize": args.font_base + 1,
            "xtick.labelsize": args.font_base - 1,
            "ytick.labelsize": args.font_base - 1,
            "legend.fontsize": args.font_base - 2,
        }
    )
    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = _pick_device(args.cuda)

    ce_W_bio, _, _ = load_connectome(args.ce_adj, args.ce_ei)
    if ce_W_bio is None:
        raise FileNotFoundError(f"Could not load CE adjacency from {args.ce_adj}")
    nnz_target_ce = int((np.abs(ce_W_bio) > 0).sum())

    if args.model == "cel":
        feature_conn = "cel"
        nnz_target = None
    else:
        feature_conn = f"er_p={args.er_p}"
        nnz_target = nnz_target_ce

    Wt, Win = build_reservoir(
        target_sr=args.rho,
        input_scale=args.input_scale,
        seed=args.seed,
        feature_conn=feature_conn,
        N=ce_W_bio.shape[0],
        ce_W_bio=ce_W_bio,
        nnz_target=nnz_target,
        DEVICE=device,
    )

    t_total = args.washout + args.t_train + args.t_test
    u = (torch.rand(t_total, 1, device=device) * 2.0 - 1.0)
    u = u - u.mean()

    X, _ = run_reservoir_with_pre(Wt, Win, u, args.leak)
    avg_signal = X.mean(dim=1, keepdim=True)

    u_m11 = to_m11(u)
    p2 = legendre_P(u_m11, 2)
    p3 = legendre_P(u_m11, 3)

    # Readouts trained on post-washout train window, predicted over post-washout timeline.
    Xtr = X[args.washout:args.washout + args.t_train]
    Ytr = torch.cat(
        [
            p2[args.washout:args.washout + args.t_train],
            p3[args.washout:args.washout + args.t_train],
        ],
        dim=1,
    )
    Xpost = X[args.washout:]
    yhat_post = ridge_fit_predict(
        Xtr,
        Ytr,
        Xpost,
        alpha=args.ridge_alpha,
        DEVICE=device,
    )
    yhat_train = yhat_post[:args.t_train]
    yhat_test = yhat_post[args.t_train:]
    p2_train = p2[args.washout:args.washout + args.t_train].squeeze(1)
    p3_train = p3[args.washout:args.washout + args.t_train].squeeze(1)
    p2_test = p2[args.washout + args.t_train:].squeeze(1)
    p3_test = p3[args.washout + args.t_train:].squeeze(1)
    p2_r2_train = r2_score(p2_train, yhat_train[:, 0])
    p3_r2_train = r2_score(p3_train, yhat_train[:, 1])
    p2_r2_test = r2_score(p2_test, yhat_test[:, 0])
    p3_r2_test = r2_score(p3_test, yhat_test[:, 1])

    t = np.arange(t_total)
    u_np = u.squeeze(1).detach().cpu().numpy()
    avg_np = avg_signal.squeeze(1).detach().cpu().numpy()
    p2_np = p2.squeeze(1).detach().cpu().numpy()
    p3_np = p3.squeeze(1).detach().cpu().numpy()
    yhat_post_np = yhat_post.detach().cpu().numpy()
    yhat_p2_np = np.full(t_total, np.nan, dtype=np.float32)
    yhat_p3_np = np.full(t_total, np.nan, dtype=np.float32)
    yhat_p2_np[args.washout:] = yhat_post_np[:, 0]
    yhat_p3_np[args.washout:] = yhat_post_np[:, 1]

    stacked = np.column_stack([t, u_np, avg_np, p2_np, p3_np, yhat_p2_np, yhat_p3_np])
    np.savetxt(
        out_csv,
        stacked,
        delimiter=",",
        header="t,input_u,avg_network_signal,legendre_p2_even,legendre_p3_odd,readout_p2,readout_p3",
        comments="",
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=args.dpi)

    t_wash_end = args.washout
    t_train_end = args.washout + args.t_train
    sec_wash_color = "#4c78a8"
    sec_train_color = "#59a14f"
    sec_test_color = "#e15759"
    read_wash_color = "#7f7fce"
    read_train_color = "#86bf7f"
    read_test_color = "#f28e8b"
    phase_bounds = [(0, t_wash_end), (t_wash_end, t_train_end), (t_train_end, t_total - 1)]
    phase_colors_target = [sec_wash_color, sec_train_color, sec_test_color]
    phase_colors_readout = [read_wash_color, read_train_color, read_test_color]

    def plot_by_phase(
        ax: plt.Axes,
        y: np.ndarray,
        phase_colors: list[str],
        *,
        linestyle: str = "-",
        linewidth: float = 1.6,
        alpha: float = 1.0,
    ) -> None:
        for (s, e), c in zip(phase_bounds, phase_colors):
            if e < s:
                continue
            idx = np.arange(s, e + 1, dtype=np.int32)
            ax.plot(t[idx], y[idx], color=c, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    plot_by_phase(axes[0], u_np, phase_colors_target, linewidth=1.8)
    axes[0].set_title("input over time")
    axes[0].set_ylabel("u(t)")

    plot_by_phase(axes[1], avg_np, phase_colors_target, linewidth=1.8)
    axes[1].set_title("average network signal over time")
    axes[1].set_ylabel("mean x(t)")

    plot_by_phase(axes[2], p2_np, phase_colors_target, linestyle="-", linewidth=1.8)
    plot_by_phase(axes[2], yhat_p2_np, phase_colors_readout, linestyle="--", linewidth=1.8, alpha=0.58)
    axes[2].set_title("legendre p even over time (use 2)")
    axes[2].set_ylabel("P2(u)")
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="target"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="readout"),
        ],
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    axes[2].text(
        0.01,
        0.95,
        f"R2 train={p2_r2_train:.3f}  test={p2_r2_test:.3f}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=args.font_small,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plot_by_phase(axes[3], p3_np, phase_colors_target, linestyle="-", linewidth=1.8)
    plot_by_phase(axes[3], yhat_p3_np, phase_colors_readout, linestyle="--", linewidth=1.8, alpha=0.58)
    axes[3].set_title("legendre p odd over time (use 3)")
    axes[3].set_ylabel("P3(u)")
    axes[3].legend(
        handles=[
            Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="target"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="readout"),
        ],
        frameon=False,
        ncol=2,
        loc="upper right",
    )
    axes[3].text(
        0.01,
        0.95,
        f"R2 train={p3_r2_train:.3f}  test={p3_r2_test:.3f}",
        transform=axes[3].transAxes,
        va="top",
        ha="left",
        fontsize=args.font_small,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
    )
    axes[3].set_xlabel("time")

    for ax in axes:
        ax.axvline(t_wash_end, color="#333333", linestyle="--", alpha=0.9, linewidth=1.4)
        ax.axvline(t_train_end, color="#333333", linestyle="--", alpha=0.9, linewidth=1.4)
        ax.grid(alpha=0.25)

    axes[0].text(
        t_wash_end,
        1.02,
        "washout ends",
        transform=axes[0].get_xaxis_transform(),
        color=sec_wash_color,
        ha="left",
        va="bottom",
        fontsize=args.font_small,
    )
    axes[0].text(
        t_train_end,
        1.02,
        "train ends / test starts",
        transform=axes[0].get_xaxis_transform(),
        color=sec_test_color,
        ha="left",
        va="bottom",
        fontsize=args.font_small,
    )

    section_handles = [
        Line2D([0], [0], color=sec_wash_color, linestyle="-", linewidth=2.0, label="washout segment"),
        Line2D([0], [0], color=sec_train_color, linestyle="-", linewidth=2.0, label="train segment"),
        Line2D([0], [0], color=sec_test_color, linestyle="-", linewidth=2.0, label="test segment"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.4, label="washout end"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.4, label="train end / test start"),
    ]
    axes[0].legend(handles=section_handles, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.34))

    model_label = "C. elegans" if args.model == "cel" else f"ER (p={args.er_p:g})"
    fig.suptitle(
        f"{model_label}: input / network / Legendre traces with readout overlays",
        y=0.998,
        fontsize=args.font_title,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
