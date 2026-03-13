#!/usr/bin/env python3
"""Generate CE-reference vs variant graph figures using project-native builders.

Data loading uses: util.util.load_connectome
Variant construction uses: util.util.build_reservoir and CE shuffle helpers
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.util import (  # noqa: E402
    _conn_and_w_shuffle_ce,
    _conn_shuffle_ce,
    _count_edges,
    _shuffle_ce_weights,
    build_reservoir,
    load_connectome,
)


@dataclass(frozen=True)
class Variant:
    key: str
    slug: str
    title: str


VARIANTS: list[Variant] = [
    Variant("ce_connectome", "01_celegans_connectome", "C. elegans connectome"),
    Variant("connection_shuffle", "02_connection_shuffle", "Connection-shuffle"),
    Variant("weight_shuffle", "03_weight_shuffle", "Weight-shuffle"),
    Variant("conn_weight_shuffle", "04_connection_then_weight_shuffle", "Connection-shuffle + weight-shuffle"),
    Variant("ce_gaussian", "05_ce_topology_gaussian_weights", "CE topology with Gaussian weights"),
    Variant("er_gaussian", "06_er_matched_gaussian", "CE-matched Erdos-Renyi with Gaussian weights"),
    Variant("ws_gaussian", "07_ws_matched_gaussian", "CE-matched Watts-Strogatz with Gaussian weights"),
    Variant("sign_gaussian_abs", "08_ce_gaussian_abs_with_original_signs", "CE + |Gaussian| with original signs"),
    Variant("sign_uniform", "09_ce_uniform_with_original_signs", "CE + Uniform with original signs"),
    Variant("sign_sampled", "10_ce_sampled_with_original_signs", "CE + sampled weights with original signs"),
    Variant("binary_local_sign", "11_ce_binary_with_original_signs", "CE + binary +/-1 with original signs"),
    Variant("binary_global_sign_shuffle", "12_ce_binary_sign_balance_shuffled", "CE + binary with shuffled signs (balance kept)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CE-reference vs variant weighted graph figures.")
    p.add_argument("--outdir", default="architecture_variant_figures/graph_examples", help="Output directory for PNGs.")
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy", help="Path to CE adjacency/weight matrix (.npy).")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy", help="Path to CE E/I labels (.npy).")
    p.add_argument("--seed", type=int, default=7, help="Base random seed.")
    p.add_argument(
        "--max-edges",
        type=int,
        default=0,
        help="Number of strongest edges to draw per panel. Use 0 to draw all nonzero edges.",
    )
    p.add_argument("--layout-iters", type=int, default=150, help="Fallback spring-layout iterations.")
    p.add_argument(
        "--layout-scale",
        type=float,
        default=1.35,
        help="Multiply node coordinates by this factor to spread nodes out.",
    )
    p.add_argument("--show-node-labels", action="store_true", help="Draw neuron-name labels on nodes.")
    p.add_argument("--label-fontsize", type=int, default=5, help="Font size for node labels when enabled.")
    p.add_argument(
        "--show-direction",
        action="store_true",
        help="Draw arrowheads to indicate edge direction (can be visually dense).",
    )
    p.add_argument(
        "--truncate-drops-negatives",
        action="store_true",
        help="If set, truncation by --max-edges may drop negative edges. Default keeps all negative edges visible.",
    )
    return p.parse_args()


def load_ce_with_project_code(ce_adj: str, ce_ei: str) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    W_bio, ei_labels, name2idx = load_connectome(ce_adj, ce_ei)
    if W_bio is None:
        raise RuntimeError("Could not load CE adjacency via util.util.load_connectome.")

    labels = None
    if name2idx is not None:
        labels = [""] * W_bio.shape[0]
        for name, idx in name2idx.items():
            labels[int(idx)] = str(name)

    return W_bio.astype(np.float32), None if ei_labels is None else ei_labels.astype(np.float32), labels


def _build_from_feature_conn(feature_conn: str, ce_W_bio: np.ndarray, seed: int) -> np.ndarray:
    nnz_target = None
    if feature_conn.startswith("er_p=") or feature_conn.startswith("ws_p="):
        nnz_target = _count_edges(ce_W_bio)

    Wt, _Win = build_reservoir(
        target_sr=None,
        input_scale=1.0,
        seed=seed,
        feature_conn=feature_conn,
        N=ce_W_bio.shape[0],
        ce_W_bio=ce_W_bio,
        drive_idx=None,
        nnz_target=nnz_target,
        DEVICE=torch.device("cpu"),
    )
    return Wt.detach().cpu().numpy().astype(np.float32)


def construct_w_with_project_code(key: str, ce_W_bio: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if key == "ce_connectome":
        return ce_W_bio.copy().astype(np.float32)

    if key == "connection_shuffle":
        return _conn_shuffle_ce(ce_W_bio, rng).astype(np.float32)

    if key == "weight_shuffle":
        return _shuffle_ce_weights(ce_W_bio, rng).astype(np.float32)

    if key == "conn_weight_shuffle":
        return _conn_and_w_shuffle_ce(ce_W_bio, rng).astype(np.float32)

    if key == "ce_gaussian":
        return _build_from_feature_conn("cel_randN", ce_W_bio, seed)

    if key == "er_gaussian":
        return _build_from_feature_conn("er_p=0.1", ce_W_bio, seed)

    if key == "ws_gaussian":
        return _build_from_feature_conn("ws_p=0.1", ce_W_bio, seed)

    if key == "sign_gaussian_abs":
        return _build_from_feature_conn("local_sign", ce_W_bio, seed)

    if key == "sign_uniform":
        return _build_from_feature_conn("local_sign+flat", ce_W_bio, seed)

    if key == "sign_sampled":
        return _build_from_feature_conn("local_sign+sample", ce_W_bio, seed)

    if key == "binary_local_sign":
        return _build_from_feature_conn("local_sign+binary", ce_W_bio, seed)

    if key == "binary_global_sign_shuffle":
        return _build_from_feature_conn("global_sign_pres", ce_W_bio, seed)

    raise ValueError(f"Unknown variant key: {key}")






def _edge_subset(
    W: np.ndarray,
    max_edges: int,
    *,
    keep_all_negatives: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz = np.argwhere(W != 0)
    if nz.size == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    w = W[nz[:, 0], nz[:, 1]].astype(np.float32)
    absw = np.abs(w)
    if max_edges <= 0 or max_edges >= nz.shape[0]:
        keep = np.arange(nz.shape[0])
    else:
        top = np.argsort(absw)[::-1][:max_edges]
        if keep_all_negatives:
            neg = np.where(w < 0)[0]
            keep = np.unique(np.concatenate([top, neg]))
        else:
            keep = top
    return nz[keep], w[keep], absw[keep]


def build_dual_colormap_norms(values: np.ndarray):
    """Build separate positive/negative color scales from variant weights."""
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)

    pos_vals = values[values > 0]
    neg_vals = values[values < 0]

    pos = None
    if pos_vals.size:
        pmax = float(pos_vals.max())
        pmin = 0.0
        if np.isclose(pmin, pmax):
            pmax = pmin + 1e-6
        pnorm = mcolors.Normalize(vmin=pmin, vmax=pmax)
        pcmap = LinearSegmentedColormap.from_list(
            "pos_saturated",
            ["#ff7f0e", "#c73e1d", "#8a1538", "#4a001f"],
            N=256,
        )
        pticks = [pmin, pmax]
        pticks = list(dict.fromkeys([float(t) for t in pticks]))
        pos = {"norm": pnorm, "cmap": pcmap, "ticks": pticks, "vmin": pmin, "vmax": pmax}

    neg = None
    if neg_vals.size:
        nmin = float(neg_vals.min())  # most negative
        nmax = 0.0
        if nmax < nmin:
            nmax = nmin
        if np.isclose(nmin, nmax):
            nmin = nmax - 1e-6
        nnorm = mcolors.Normalize(vmin=nmin, vmax=nmax)
        ncmap = LinearSegmentedColormap.from_list(
            "neg_saturated",
            ["#001f54", "#003f88", "#006daa", "#0096c7"],
            N=256,
        )
        nticks = [nmin, nmax]
        nticks = list(dict.fromkeys([float(t) for t in nticks]))
        neg = {"norm": nnorm, "cmap": ncmap, "ticks": nticks, "vmin": nmin, "vmax": nmax}

    return pos, neg


def draw_weighted_panel(
    ax: plt.Axes,
    W: np.ndarray,
    pos: dict[int, np.ndarray],
    title: str,
    node_colors: list[str],
    node_labels: list[str] | None,
    label_fontsize: int,
    node_size: int,
    max_edges: int,
    keep_all_negatives: bool,
    show_direction: bool,
    pos_norm: mcolors.Normalize | None,
    pos_cmap,
    neg_norm: mcolors.Normalize | None,
    neg_cmap,
) -> None:
    n = W.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    edge_idx, w, _absw = _edge_subset(W, max_edges=max_edges, keep_all_negatives=keep_all_negatives)
    for (i, j), wij in zip(edge_idx, w):
        G.add_edge(int(i), int(j), weight=float(wij))

    ax.set_title(title, fontsize=11)
    ax.axis("off")

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="#2b2b2b",
        linewidths=0.25,
        ax=ax,
    )
    if node_labels is not None:
        lbl_map = {i: node_labels[i] for i in range(min(len(node_labels), n)) if node_labels[i]}
        if lbl_map:
            nx.draw_networkx_labels(
                G,
                pos=pos,
                labels=lbl_map,
                font_size=label_fontsize,
                font_color="#1f1f1f",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.1},
                ax=ax,
            )

    if w.size == 0:
        return

    pos_edges = [(int(i), int(j)) for (i, j), wij in zip(edge_idx, w) if wij >= 0]
    pos_vals = [float(wij) for wij in w if wij >= 0]
    neg_edges = [(int(i), int(j)) for (i, j), wij in zip(edge_idx, w) if wij < 0]
    neg_vals = [float(wij) for wij in w if wij < 0]

    abs_max = max(float(np.max(np.abs(w))), 1e-9)
    edge_kwargs = {}
    if show_direction:
        edge_kwargs = {
            "arrows": True,
            "arrowsize": 7,
            "arrowstyle": "-|>",
            "connectionstyle": "arc3,rad=0.08",
        }
    else:
        edge_kwargs = {"arrows": False}

    if pos_edges:
        pos_kwargs = {}
        if pos_norm is not None and pos_cmap is not None:
            pos_kwargs = {
                "edge_color": pos_vals,
                "edge_cmap": pos_cmap,
                "edge_vmin": pos_norm.vmin,
                "edge_vmax": pos_norm.vmax,
            }
        else:
            pos_kwargs = {"edge_color": "#d7301f"}
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=pos_edges,
            width=[0.55 + 2.2 * (abs(v) / 37) for v in pos_vals],
            alpha=0.40,
            **pos_kwargs,
            **edge_kwargs,
        )

    if neg_edges:
        neg_kwargs = {}
        if neg_norm is not None and neg_cmap is not None:
            neg_kwargs = {
                "edge_color": neg_vals,
                "edge_cmap": neg_cmap,
                "edge_vmin": neg_norm.vmin,
                "edge_vmax": neg_norm.vmax,
            }
        else:
            print(len(G))
            neg_kwargs = {"edge_color": "#2166ac"}
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=neg_edges,
            width=[0.55 + 2.2 * (abs(v) / 37) for v in neg_vals],
            alpha=0.70 if (len(neg_edges) < (((len(pos_edges)+len(neg_edges))// 2) - 50)) else 0.30,
            **neg_kwargs,
            **edge_kwargs,
        )


def stats_line(W: np.ndarray) -> str:
    nnz = int((W != 0).sum())
    pos = int((W > 0).sum())
    neg = int((W < 0).sum())
    return f"N={W.shape[0]}, edges={nnz}, +={pos}, -={neg}"


def compute_ce_kamada_layout(
    W: np.ndarray,
    seed: int,
    fallback_iters: int,
    layout_scale: float,
) -> dict[int, np.ndarray]:
    n = W.shape[0]
    A = np.abs(W)

    H = nx.Graph()
    H.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            mag = max(float(A[i, j]), float(A[j, i]))
            if mag > 0.0:
                H.add_edge(i, j, length=1.0 / max(mag, 1e-9))

    pos = nx.kamada_kawai_layout(H, weight="length")


    s = float(layout_scale) if layout_scale > 0 else 1.0
    return {k: np.asarray(v) * s for k, v in pos.items()}


def make_variant_figure(
    variant: Variant,
    ce_base: np.ndarray,
    ce_pos: dict[int, np.ndarray],
    ce_ei: np.ndarray | None,
    ce_labels: list[str] | None,
    outdir: Path,
    seed: int,
    max_edges: int,
    keep_all_negatives: bool,
    show_node_labels: bool,
    label_fontsize: int,
    show_direction: bool,
) -> None:
    W_ref = ce_base
    W_var = construct_w_with_project_code(variant.key, ce_base, seed + 20_000)

    _, ref_w, _ = _edge_subset(W_ref, max_edges=max_edges, keep_all_negatives=keep_all_negatives)
    _var_edges, _var_w, _ = _edge_subset(W_var, max_edges=max_edges, keep_all_negatives=keep_all_negatives)
    scale_values = ref_w
    pos_map, neg_map = build_dual_colormap_norms(scale_values)

    fig = plt.figure(figsize=(17.4, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.09], wspace=0.04)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    cbar_gs = gs[0, 2].subgridspec(2, 1, hspace=0.35)
    cax_pos = fig.add_subplot(cbar_gs[0, 0])
    cax_neg = fig.add_subplot(cbar_gs[1, 0])

    draw_weighted_panel(
        ax_left,
        W_ref,
        ce_pos,
        title=f"Reference: Biological C. elegans ({stats_line(W_ref)})",
        node_colors=["#666666"] * W_ref.shape[0],
        node_labels=ce_labels if show_node_labels else None,
        label_fontsize=label_fontsize,
        node_size=22,
        max_edges=max_edges,
        keep_all_negatives=keep_all_negatives,
        show_direction=show_direction,
        pos_norm=None if pos_map is None else pos_map["norm"],
        pos_cmap=None if pos_map is None else pos_map["cmap"],
        neg_norm=None if neg_map is None else neg_map["norm"],
        neg_cmap=None if neg_map is None else neg_map["cmap"],
    )

    draw_weighted_panel(
        ax_right,
        W_var,
        ce_pos,
        title=f"Variant under test ({stats_line(W_var)})",
        node_colors=["#666666"] * W_var.shape[0],
        node_labels=ce_labels if show_node_labels else None,
        label_fontsize=label_fontsize,
        node_size=22,
        max_edges=max_edges,
        keep_all_negatives=keep_all_negatives,
        show_direction=show_direction,
        pos_norm=None if pos_map is None else pos_map["norm"],
        pos_cmap=None if pos_map is None else pos_map["cmap"],
        neg_norm=None if neg_map is None else neg_map["norm"],
        neg_cmap=None if neg_map is None else neg_map["cmap"],
    )

    if pos_map is not None:
        sm_pos = plt.cm.ScalarMappable(norm=pos_map["norm"], cmap=pos_map["cmap"])
        sm_pos.set_array([])
        cb_pos = fig.colorbar(sm_pos, cax=cax_pos)
        cb_pos.set_ticks(pos_map["ticks"])
        cb_pos.ax.set_title("Positive", fontsize=9, pad=6)
        cb_pos.set_label("w (+)", fontsize=9)
        cb_pos.ax.tick_params(labelsize=8)
    else:
        cax_pos.axis("off")

    if neg_map is not None:
        sm_neg = plt.cm.ScalarMappable(norm=neg_map["norm"], cmap=neg_map["cmap"])
        sm_neg.set_array([])
        cb_neg = fig.colorbar(sm_neg, cax=cax_neg)
        cb_neg.set_ticks(neg_map["ticks"])
        cb_neg.ax.set_title("Negative", fontsize=9, pad=6)
        cb_neg.set_label("w (-)", fontsize=9)
        cb_neg.ax.tick_params(labelsize=8)
    else:
        cax_neg.axis("off")

    style_handles = [
        Line2D([0], [0], color="#c73e1d", lw=2.0, ls="-", label="positive edge"),
        Line2D([0], [0], color="#006daa", lw=2.0, ls="-", label="negative edge"),
    ]
    fig.legend(handles=style_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=10)

    fig.suptitle(f"{variant.slug}: {variant.title}", fontsize=15, y=1.02)
    fig.subplots_adjust(left=0.02, right=0.96, top=0.90, bottom=0.04, wspace=0.05)
    fig.savefig(outdir / f"{variant.slug}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def make_index_figure(outdir: Path) -> None:
    fig = plt.figure(figsize=(14, 3.1))
    ax = fig.add_subplot(111)
    ax.axis("off")
    lines = [
        "Architecture variant comparison files (left = CE reference, right = variant):",
        "Data loading + W construction use project-native util code.",
    ]
    for v in VARIANTS:
        lines.append(f"{v.slug}.png  -  {v.title}")
    ax.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "00_file_index.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ce_base, ce_ei, ce_labels = load_ce_with_project_code(args.ce_adj, args.ce_ei)
    ce_pos = compute_ce_kamada_layout(
        ce_base,
        seed=args.seed,
        fallback_iters=args.layout_iters,
        layout_scale=args.layout_scale,
    )
    keep_all_negatives = not args.truncate_drops_negatives

    make_index_figure(outdir)
    for i, variant in enumerate(VARIANTS):
        make_variant_figure(
            variant=variant,
            ce_base=ce_base,
            ce_pos=ce_pos,
            ce_ei=ce_ei,
            ce_labels=ce_labels,
            outdir=outdir,
            seed=args.seed + i * 97,
            max_edges=args.max_edges,
            keep_all_negatives=keep_all_negatives,
            show_node_labels=args.show_node_labels,
            label_fontsize=args.label_fontsize,
            show_direction=args.show_direction,
        )

    print(f"Wrote {1 + len(VARIANTS)} files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
