#!/usr/bin/env python3
"""Animate input driving the reservoir and the corresponding readout trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import torch

from inv_arc_test import _pick_device
from network_stats.run_one import run_reservoir_with_pre
from network_stats.stats_util import legendre_P, ridge_fit_predict
from util.util import build_reservoir, load_connectome, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a network/readout animation.")
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy")
    p.add_argument("--out-mp4", default="ipc_order_sweep_ce_many/network_readout_animation.mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho", type=float, default=0.95)
    p.add_argument("--leak", type=float, default=0.8)
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--washout", type=int, default=12)
    p.add_argument("--t-train", type=int, default=40)
    p.add_argument("--t-test", type=int, default=36)
    p.add_argument("--delay", type=int, default=12)
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max-edges", type=int, default=2200)
    p.add_argument("--cuda", type=int, default=None)
    return p.parse_args()


def normalize_pos(pos: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    pts = np.array(list(pos.values()), dtype=np.float32)
    center = pts.mean(axis=0)
    span = np.abs(pts - center).max()
    return {k: (np.asarray(v, dtype=np.float32) - center) / (span + 1e-9) for k, v in pos.items()}


def make_layout(W_bio: np.ndarray, max_edges: int, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    edges = np.argwhere(np.abs(W_bio) > 0)
    rng = np.random.default_rng(seed)
    if len(edges) > max_edges:
        edges = edges[rng.choice(len(edges), size=max_edges, replace=False)]
    edge_pairs = [(int(i), int(j)) for i, j in edges if i != j]

    graph = nx.Graph()
    graph.add_nodes_from(range(W_bio.shape[0]))
    graph.add_edges_from((i, j) for i, j in edge_pairs)
    degree = np.array([graph.degree(i) for i in range(W_bio.shape[0])])
    order = np.argsort(-degree)
    xy = np.zeros((W_bio.shape[0], 2), dtype=np.float32)
    for rank, node in enumerate(order):
        theta = 2.0 * np.pi * rank / W_bio.shape[0]
        radius = 0.48 + 0.50 * (rank / max(W_bio.shape[0] - 1, 1))
        xy[node] = [radius * np.cos(theta), radius * np.sin(theta)]
    return xy, edge_pairs


def main() -> None:
    args = parse_args()
    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = _pick_device(args.cuda)

    ce_W_bio, _, _ = load_connectome(args.ce_adj, args.ce_ei)
    if ce_W_bio is None:
        raise FileNotFoundError(f"Could not load CE adjacency from {args.ce_adj}")

    Wt, Win = build_reservoir(
        target_sr=args.rho,
        input_scale=args.input_scale,
        seed=args.seed,
        feature_conn="cel",
        N=ce_W_bio.shape[0],
        ce_W_bio=ce_W_bio,
        nnz_target=None,
        DEVICE=device,
    )

    t_total = args.washout + args.t_train + args.t_test
    sim_total = t_total + args.delay
    u = torch.rand(sim_total, 1, device=device) * 2.0 - 1.0
    X, _ = run_reservoir_with_pre(Wt, Win, u, args.leak)
    p3 = legendre_P(u, 3).squeeze(1).detach().cpu().numpy()
    X = X[args.delay :]
    true_signal = p3[:t_total].astype(np.float32)

    Xtr = X[args.washout : args.washout + args.t_train]
    ytr_np = true_signal[args.washout : args.washout + args.t_train]
    if not np.isfinite(ytr_np).all():
        raise ValueError("Delay is too large for the current washout/training split.")
    ytr = torch.as_tensor(ytr_np, dtype=X.dtype, device=device).unsqueeze(1)
    yhat_post = ridge_fit_predict(Xtr, ytr, X[args.washout :], alpha=args.ridge_alpha, DEVICE=device)

    t = np.arange(t_total, dtype=np.float32)
    X_np = X.detach().cpu().numpy()
    prediction = np.full(t_total, np.nan, dtype=np.float32)
    prediction[args.washout :] = yhat_post.squeeze(1).detach().cpu().numpy()

    xy, edge_pairs = make_layout(ce_W_bio, args.max_edges, args.seed)
    network_segments = np.array([(xy[i], xy[j]) for i, j in edge_pairs], dtype=np.float32)
    network_xy = xy * 0.86
    network_xy[:, 0] += 0.04
    input_xy = np.array([-1.55, 0.0], dtype=np.float32)
    readout_xy = np.array([1.55, 0.0], dtype=np.float32)
    input_segments = np.array([(input_xy, p) for p in network_xy], dtype=np.float32)
    readout_segments = np.array([(p, readout_xy) for p in network_xy], dtype=np.float32)
    network_segments = np.array([(network_xy[i], network_xy[j]) for i, j in edge_pairs], dtype=np.float32)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.02)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[1, 0])

    ax_net.set_aspect("equal")
    ax_net.set_xlim(-1.78, 1.78)
    ax_net.set_ylim(-0.88, 1.00)
    ax_net.axis("off")
    ax_net.text(input_xy[0], 0.96, "true past signal", ha="center", va="center", fontsize=14)
    ax_net.text(0.04, 0.96, "C. elegans reservoir", ha="center", va="center", fontsize=14)
    ax_net.text(readout_xy[0], 0.96, "readout", ha="center", va="center", fontsize=14)
    ax_net.annotate(
        "",
        xy=(-1.05, 0.0),
        xytext=(-1.38, 0.0),
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.8, alpha=0.85),
    )
    ax_net.annotate(
        "",
        xy=(1.38, 0.0),
        xytext=(1.05, 0.0),
        arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.8, alpha=0.85),
    )
    ax_net.add_collection(LineCollection(network_segments, colors="#1f1f1f", linewidths=0.28, alpha=0.20))
    input_edges = LineCollection(input_segments, colors="#2f6f9f", linewidths=0.32, alpha=0.10)
    ax_net.add_collection(input_edges)
    readout_edges = LineCollection(readout_segments, colors="#d84a3a", linewidths=0.32, alpha=0.05)
    ax_net.add_collection(readout_edges)
    nodes = ax_net.scatter(
        network_xy[:, 0],
        network_xy[:, 1],
        c=X_np[0],
        cmap="coolwarm",
        vmin=-0.9,
        vmax=0.9,
        s=30,
        linewidths=0.14,
        edgecolors="#333333",
        alpha=0.9,
    )
    first_true = 0.0 if not np.isfinite(true_signal[0]) else float(true_signal[0])
    input_node = ax_net.scatter(
        [input_xy[0]],
        [input_xy[1]],
        c=[first_true],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=260,
        edgecolors="#222222",
        linewidths=1.2,
        zorder=3,
    )
    readout_node = ax_net.scatter(
        [readout_xy[0]],
        [readout_xy[1]],
        c=[0.0],
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=260,
        edgecolors="#222222",
        linewidths=1.2,
        zorder=3,
    )

    washout_end = float(args.washout)
    test_start = float(args.washout + args.t_train)
    ax_plot.axvspan(t[0], washout_end, color="#4c78a8", alpha=0.08)
    ax_plot.axvspan(washout_end, test_start, color="#59a14f", alpha=0.08)
    ax_plot.axvspan(test_start, t[-1], color="#e15759", alpha=0.08)
    ax_plot.axvline(washout_end, ymin=0.22, ymax=0.98, color="#555555", linewidth=1.2, linestyle="--", alpha=0.5)
    ax_plot.axvline(test_start, color="#555555", linewidth=1.2, linestyle="--", alpha=0.5)
    ax_plot.text((t[0] + washout_end) / 2, 0.94, "washout", ha="center", va="top", transform=ax_plot.get_xaxis_transform())
    ax_plot.text((washout_end + test_start) / 2, 0.94, "training", ha="center", va="top", transform=ax_plot.get_xaxis_transform())
    ax_plot.text((test_start + t[-1]) / 2, 0.94, "prediction", ha="center", va="top", transform=ax_plot.get_xaxis_transform())
    ax_plot.set_xlim(t[0] - 1, t[-1] + 1)
    ax_plot.set_ylim(-1.1, 1.1)
    ax_plot.set_xlabel("time")
    ax_plot.set_ylabel("signal")
    ax_plot.grid(alpha=0.2, linewidth=0.8)
    true_line, = ax_plot.plot([], [], color="#2f6f9f", linewidth=2.8, alpha=0.78, label="true past signal")
    pred_line, = ax_plot.plot([], [], color="#d84a3a", linewidth=2.8, alpha=0.78, label="readout prediction")
    true_dot, = ax_plot.plot([], [], marker="o", color="#2f6f9f", markersize=6, linestyle="None", alpha=0.9)
    pred_dot, = ax_plot.plot([], [], marker="o", color="#d84a3a", markersize=6, linestyle="None", alpha=0.9)
    cursor = ax_plot.axvline(t[0], color="#111111", linewidth=1.1, alpha=0.55)
    ax_plot.legend(frameon=False, loc="lower left", ncol=2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.09)

    def update(frame: int):
        nodes.set_array(X_np[frame])
        current_true = 0.0 if not np.isfinite(true_signal[frame]) else float(true_signal[frame])
        input_node.set_array(np.array([current_true], dtype=np.float32))
        if np.isfinite(prediction[frame]):
            readout_node.set_array(np.array([prediction[frame]], dtype=np.float32))
            readout_edges.set_alpha(0.06 + 0.22 * abs(float(prediction[frame])))
        else:
            readout_node.set_array(np.array([0.0], dtype=np.float32))
            readout_edges.set_alpha(0.03)
        input_edges.set_alpha(0.08 + 0.24 * abs(current_true))
        true_line.set_data(t[: frame + 1], true_signal[: frame + 1])
        if np.isfinite(true_signal[frame]):
            true_dot.set_data([t[frame]], [true_signal[frame]])
        else:
            true_dot.set_data([], [])
        pred_idx = np.isfinite(prediction[: frame + 1])
        pred_line.set_data(t[: frame + 1][pred_idx], prediction[: frame + 1][pred_idx])
        if np.isfinite(prediction[frame]):
            pred_dot.set_data([t[frame]], [prediction[frame]])
        else:
            pred_dot.set_data([], [])
        cursor.set_xdata([t[frame], t[frame]])
        return nodes, input_node, readout_node, input_edges, readout_edges, true_line, pred_line, true_dot, pred_dot, cursor

    anim = FuncAnimation(fig, update, frames=t_total, interval=1000 / args.fps, blit=False)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=2200,
        extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"],
    )
    anim.save(out_mp4, writer=writer)
    plt.close(fig)
    print(f"[done] wrote {out_mp4}")


if __name__ == "__main__":
    main()
