#!/usr/bin/env python3
"""Generate CE-reference vs variant graph figures using project-native builders.

Data loading uses: util.util.load_connectome
Variant construction uses: util.util.build_reservoir and CE shuffle helpers
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
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
    _sample_from_cel,
    _shuffle_ce_weights,
    build_reservoir,
    load_connectome,
)


@dataclass(frozen=True)
class Variant:
    key: str
    slug: str
    title: str


ALL_JOB_KEYS: tuple[str, ...] = (
    "real",
    "shuffle_weights",
    "cel_randN",
    "er_randN",
    "ws_p01_randN",
    "conn_shuf",
    "local_sign",
    "conn_shuf_only",
    "cel_sample",
    "local_sign+flat",
    "local_sign+sample",
    "local_sign+binary",
    "global_sign_pres",
    "binary_base",
    "binary_base_topology_shuffle",
    "binary+shuffle",
    "binary+conshuffle+wshuffle",
)

JOB_TITLES: dict[str, str] = {
    "real": "C. elegans connectome (real)",
    "shuffle_weights": "Weight-shuffle",
    "cel_randN": "CE topology + Gaussian weights",
    "er_randN": "ER (p=0.1) + Gaussian weights",
    "ws_p01_randN": "WS (p=0.1) + Gaussian weights",
    "conn_shuf": "Connection-shuffle + weight-shuffle",
    "local_sign": "Local sign-preserved Gaussian magnitudes",
    "conn_shuf_only": "Connection-shuffle only",
    "cel_sample": "CE topology + sampled CE weights",
    "local_sign+flat": "Local sign-preserved Uniform magnitudes",
    "local_sign+sample": "Local sign-preserved sampled magnitudes",
    "local_sign+binary": "Local sign-preserved binary magnitudes",
    "global_sign_pres": "Binary weights + global sign-balance shuffle",
    "binary_base": "Unsigned binary base (CE topology)",
    "binary_base_topology_shuffle": "Unsigned binary + topology shuffle",
    "binary+shuffle": "Binary + connection shuffle",
    "binary+conshuffle+wshuffle": "Binary + connection shuffle + weight shuffle",
}

JOB_SLUG_OVERRIDES: dict[str, str] = {
    "real": "celegans_connectome",
    "shuffle_weights": "weight_shuffle",
    "cel_randN": "ce_gaussian_weights",
    "er_randN": "er_p01_gaussian_weights",
    "ws_p01_randN": "ws_p01_gaussian_weights",
    "conn_shuf": "connection_and_weight_shuffle",
    "local_sign": "local_sign_gaussian_abs",
    "conn_shuf_only": "connection_shuffle_only",
    "cel_sample": "ce_sampled_weights",
    "local_sign+flat": "local_sign_uniform",
    "local_sign+sample": "local_sign_sampled",
    "local_sign+binary": "local_sign_binary",
    "global_sign_pres": "global_sign_preserved_binary",
    "binary_base": "binary_base",
    "binary_base_topology_shuffle": "binary_base_topology_shuffle",
    "binary+shuffle": "binary_plus_shuffle",
    "binary+conshuffle+wshuffle": "binary_plus_connshuffle_plus_wshuffle",
}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _build_variants() -> list[Variant]:
    variants: list[Variant] = []
    for idx, key in enumerate(ALL_JOB_KEYS, start=1):
        base_slug = JOB_SLUG_OVERRIDES.get(key,_slugify(key))
        variants.append(Variant(key=key, slug=f"{base_slug}", title=JOB_TITLES.get(key, key)))
    return variants


VARIANTS: list[Variant] = _build_variants()

# Variants that alter the graph's connectivity (edge locations), not just edge weights.
CONNECTIVITY_ALTERING_KEYS: frozenset[str] = frozenset(
    {
        "er_randN",
        "ws_p01_randN",
        "conn_shuf",
        "conn_shuf_only",
        "binary_base_topology_shuffle",
        "binary+shuffle",
        "binary+conshuffle+wshuffle",
    }
)


def variant_alters_connectivity(key: str) -> bool:
    return key in CONNECTIVITY_ALTERING_KEYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CE-reference vs variant weighted graph figures.")
    p.add_argument("--outdir", default="architecture_variant_figures/graph_examples", help="Output directory for PNGs.")
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy", help="Path to CE adjacency/weight matrix (.npy).")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy", help="Path to CE E/I labels (.npy).")
    p.add_argument("--seed", type=int, default=7, help="Base random seed.")
    p.add_argument("--er-p", type=float, default=0.1, help="ER probability for er_randN.")
    p.add_argument("--ws-p", type=float, default=0.1, help="WS rewiring probability for ws_p01_randN.")
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
        default=1.5,
        help="Multiply node coordinates by this factor to spread nodes out.",
    )
    p.add_argument("--show-node-labels", action="store_true", help="Draw neuron-name labels on nodes.")
    p.add_argument("--label-fontsize", type=int, default=10, help="Font size for node labels when enabled.")
    p.add_argument("--panel-title-fontsize", type=int, default=28, help="Panel title font size.")
    p.add_argument("--suptitle-fontsize", type=int, default=36, help="Figure super-title font size.")
    p.add_argument("--legend-fontsize", type=int, default=24, help="Legend font size.")
    p.add_argument("--cbar-title-fontsize", type=int, default=28, help="Colorbar title font size.")
    p.add_argument("--cbar-label-fontsize", type=int, default=24, help="Colorbar label font size.")
    p.add_argument("--cbar-tick-fontsize", type=int, default=24, help="Colorbar tick font size.")
    p.add_argument("--index-fontsize", type=int, default=24, help="Index page font size.")
    p.add_argument("--figure-dpi", type=int, default=320, help="Output DPI.")
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


def _ce_negative_fraction(W: np.ndarray) -> float:
    nz = int((W != 0).sum())
    if nz == 0:
        return 0.0
    return float((W < 0).sum()) / float(nz)


def _build_from_feature_conn(
    feature_conn: str,
    ce_W_bio: np.ndarray,
    seed: int,
    *,
    per_neg: float | None = None,
    alpha: float | None = None,
) -> np.ndarray:
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
        per_neg=per_neg,
        alpha=alpha,
    )
    return Wt.detach().cpu().numpy().astype(np.float32)


def construct_w_with_project_code(
    key: str,
    ce_W_bio: np.ndarray,
    seed: int,
    er_p: float,
    ws_p: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if key == "real":
        return ce_W_bio.copy().astype(np.float32)

    if key == "shuffle_weights":
        return _shuffle_ce_weights(ce_W_bio, rng).astype(np.float32)

    if key == "cel_randN":
        return _build_from_feature_conn("cel_randN", ce_W_bio, seed)

    if key == "er_randN":
        return _build_from_feature_conn(f"er_p={er_p}", ce_W_bio, seed)

    if key == "ws_p01_randN":
        return _build_from_feature_conn(f"ws_p={ws_p}", ce_W_bio, seed)

    if key == "conn_shuf":
        return _conn_and_w_shuffle_ce(ce_W_bio, rng).astype(np.float32)

    if key == "local_sign":
        return _build_from_feature_conn("local_sign", ce_W_bio, seed)

    if key == "conn_shuf_only":
        return _conn_shuffle_ce(ce_W_bio, rng).astype(np.float32)

    if key == "cel_sample":
        return _sample_from_cel(ce_W_bio, rng).astype(np.float32)

    if key == "local_sign+flat":
        return _build_from_feature_conn("local_sign+flat", ce_W_bio, seed)

    if key == "local_sign+sample":
        return _build_from_feature_conn("local_sign+sample", ce_W_bio, seed)

    if key == "local_sign+binary":
        return _build_from_feature_conn("local_sign+binary", ce_W_bio, seed)

    if key == "global_sign_pres":
        return _build_from_feature_conn("global_sign_pres", ce_W_bio, seed)

    if key == "binary_base":
        return _build_from_feature_conn("binary_base", ce_W_bio, seed)

    if key == "binary_base_topology_shuffle":
        return _build_from_feature_conn("binary_base_topology_shuffle", ce_W_bio, seed)

    if key == "binary+shuffle":
        return _build_from_feature_conn("binary+shuffle", ce_W_bio, seed)

    if key == "binary+conshuffle+wshuffle":
        return _build_from_feature_conn("binary+conshuffle+wshuffle", ce_W_bio, seed)

    if key == "sign_test_og_cel":
        return _build_from_feature_conn(
            "sign_test_og_cel",
            ce_W_bio,
            seed,
            per_neg=_ce_negative_fraction(ce_W_bio),
        )

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
    panel_title_fontsize: int,
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

    coords = np.array([pos[i] for i in range(n) if i in pos], dtype=np.float32)
    if coords.size:
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        xspan = float(max(xmax - xmin, 1e-6))
        yspan = float(max(ymax - ymin, 1e-6))
        pad_frac = 0.01
        ax.set_xlim(float(xmin - pad_frac * xspan), float(xmax + pad_frac * xspan))
        ax.set_ylim(float(ymin - pad_frac * yspan), float(ymax + pad_frac * yspan))
        # Fill the panel area instead of forcing a square plotting box.
        ax.set_aspect("auto")

    if w.size == 0:
        return

    pos_edges = [(int(i), int(j)) for (i, j), wij in zip(edge_idx, w) if wij >= 0]
    pos_vals = [float(wij) for wij in w if wij >= 0]
    neg_edges = [(int(i), int(j)) for (i, j), wij in zip(edge_idx, w) if wij < 0]
    neg_vals = [float(wij) for wij in w if wij < 0]

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
    return f"+={pos}, -={neg}"


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
    ce_labels: list[str] | None,
    outdir: Path,
    seed: int,
    er_p: float,
    ws_p: float,
    max_edges: int,
    layout_iters: int,
    layout_scale: float,
    keep_all_negatives: bool,
    show_node_labels: bool,
    label_fontsize: int,
    panel_title_fontsize: int,
    suptitle_fontsize: int,
    legend_fontsize: int,
    cbar_title_fontsize: int,
    cbar_label_fontsize: int,
    cbar_tick_fontsize: int,
    show_direction: bool,
    figure_dpi: int,
) -> None:
    W_ref = ce_base
    W_var = construct_w_with_project_code(variant.key, ce_base, seed + 20_000, er_p=er_p, ws_p=ws_p)
    panel_pos = ce_pos
    if variant_alters_connectivity(variant.key):
        panel_pos = compute_ce_kamada_layout(
            W_var,
            seed=seed + 40_000,
            fallback_iters=layout_iters,
            layout_scale=layout_scale,
        )

    _, ref_w, _ = _edge_subset(W_ref, max_edges=max_edges, keep_all_negatives=keep_all_negatives)
    _var_edges, _var_w, _ = _edge_subset(W_var, max_edges=max_edges, keep_all_negatives=keep_all_negatives)
    scale_values = ref_w
    pos_map, neg_map = build_dual_colormap_norms(scale_values)

    fig = plt.figure(figsize=(11.5, 9.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.06], wspace=0.03)
    ax_right =  fig.add_subplot(gs[0, 0])
    cbar_gs = gs[0, 1].subgridspec(2, 1, hspace=0.35)
    cax_pos = fig.add_subplot(cbar_gs[0, 0])
    cax_neg = fig.add_subplot(cbar_gs[1, 0])



    draw_weighted_panel(
        ax_right,
        W_var,
        panel_pos,
        title=f"{variant.slug} ({stats_line(W_var)})",
        panel_title_fontsize=panel_title_fontsize,
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


    cax_pos.axis("off")

    cax_neg.axis("off")



    #fig.suptitle(f"{variant.slug}: {variant.title}", fontsize=suptitle_fontsize, y=1.02)
    fig.subplots_adjust(left=0.00, right=1.0, top=1.0, bottom=0.00, wspace=0.00)
    fig.savefig(outdir / f"{variant.slug}.png", dpi=figure_dpi)
    plt.close(fig)


def make_index_figure(outdir: Path, index_fontsize: int, figure_dpi: int) -> None:
    lines = [
        "Architecture variant graph files (single-panel per model):",
        "Data loading + W construction use project-native util code.",
    ]
    for v in VARIANTS:
        lines.append(f"{v.slug}.png  -  {v.title}")

    fig_h = max(4.0, 0.62 * len(lines))
    fig = plt.figure(figsize=(14, fig_h))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=index_fontsize)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(outdir / "00_file_index.png", dpi=figure_dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ce_base, _ce_ei, ce_labels = load_ce_with_project_code(args.ce_adj, args.ce_ei)
    ce_pos = compute_ce_kamada_layout(
        ce_base,
        seed=args.seed,
        fallback_iters=args.layout_iters,
        layout_scale=args.layout_scale,
    )
    keep_all_negatives = not args.truncate_drops_negatives

    make_index_figure(outdir, index_fontsize=args.index_fontsize, figure_dpi=args.figure_dpi)
    for i, variant in enumerate(VARIANTS):
        make_variant_figure(
            variant=variant,
            ce_base=ce_base,
            ce_pos=ce_pos,
            ce_labels=ce_labels,
            outdir=outdir,
            seed=args.seed + i * 97,
            er_p=args.er_p,
            ws_p=args.ws_p,
            max_edges=args.max_edges,
            layout_iters=args.layout_iters,
            layout_scale=args.layout_scale,
            keep_all_negatives=keep_all_negatives,
            show_node_labels=args.show_node_labels,
            label_fontsize=args.label_fontsize,
            panel_title_fontsize=args.panel_title_fontsize,
            suptitle_fontsize=args.suptitle_fontsize,
            legend_fontsize=args.legend_fontsize,
            cbar_title_fontsize=args.cbar_title_fontsize,
            cbar_label_fontsize=args.cbar_label_fontsize,
            cbar_tick_fontsize=args.cbar_tick_fontsize,
            show_direction=args.show_direction,
            figure_dpi=args.figure_dpi,
        )

    print(f"Wrote {1 + len(VARIANTS)} files to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
