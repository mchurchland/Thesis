#!/usr/bin/env python3
"""Generate thesis-ready visual maps for connectome architecture variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch


PROPERTY_ORDER: list[tuple[str, str]] = [
    ("ce_topology", "CE topology base"),
    ("rewire_connections", "Connections rewired"),
    ("random_graph_model", "Random graph model"),
    ("degree_preserved", "Degree preserved"),
    ("edge_count_matched_ce", "Edge count matched to CE"),
    ("weights_preserved", "Weight values untouched"),
    ("weight_shuffle", "Weights shuffled"),
    ("gaussian_weights", "Gaussian weights"),
    ("uniform_weights", "Uniform weights"),
    ("ce_sampled_weights", "Sampled from CE weights"),
    ("binary_weights", "Binary +/-1 weights"),
    ("local_sign_preserved", "Local sign preserved"),
    ("global_sign_balance_preserved", "Global sign balance preserved"),
]


@dataclass(frozen=True)
class ArchitectureSpec:
    slug: str
    title: str
    short_label: str
    topology_step: str
    weight_step: str
    sign_step: str
    formula: str
    note: str
    props: dict[str, int]


def _props(**kwargs: int) -> dict[str, int]:
    base = {k: 0 for k, _ in PROPERTY_ORDER}
    for key, value in kwargs.items():
        if key not in base:
            raise KeyError(f"Unknown property: {key}")
        base[key] = int(bool(value))
    return base


def build_architectures() -> list[ArchitectureSpec]:
    return [
        ArchitectureSpec(
            slug="01_celegans_connectome",
            title="C. elegans connectome",
            short_label="CE connectome",
            topology_step="Use biological CE adjacency with no rewiring.",
            weight_step="Use biological CE synaptic weights as-is.",
            sign_step="Signs stay exactly as biological data.",
            formula="W_ij = W^CE_ij",
            note="Baseline biological model.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                weights_preserved=1,
                local_sign_preserved=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="02_connection_shuffle",
            title="Connection-shuffle",
            short_label="Connection-shuffle",
            topology_step="Rewire CE edges by pair swapping endpoints (a,b)+(c,d)->(a,d)+(c,b).",
            weight_step="Keep edge weights attached to moved edges.",
            sign_step="No per-edge sign lock after rewiring; global sign count is preserved.",
            formula="Degree-preserving edge swap on CE graph",
            note="Preserves in/out degree while changing neighbors.",
            props=_props(
                ce_topology=1,
                rewire_connections=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                weights_preserved=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="03_weight_shuffle",
            title="Weight-shuffle",
            short_label="Weight-shuffle",
            topology_step="Keep CE connectivity fixed.",
            weight_step="Shuffle nonzero CE weights across existing CE edges.",
            sign_step="Signs can flip on specific edges; global sign balance remains the same.",
            formula="W_nz <- shuffle(W_nz) on fixed CE support",
            note="Expected sign-flip probability from your text: 11.47%.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                weight_shuffle=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="04_connection_then_weight_shuffle",
            title="Connection-shuffle + weight-shuffle",
            short_label="Conn + weight shuffle",
            topology_step="First apply degree-preserving CE connection shuffle.",
            weight_step="Then shuffle nonzero weights across the rewired support.",
            sign_step="No per-edge sign lock; global sign balance remains the same.",
            formula="ConnShuffle(CE) then WeightShuffle(nonzero)",
            note="Control for added weight randomization after rewiring.",
            props=_props(
                ce_topology=1,
                rewire_connections=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                weight_shuffle=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="05_ce_topology_gaussian_weights",
            title="CE topology with Gaussian weights",
            short_label="CE + Gaussian",
            topology_step="Keep CE adjacency (same nonzero support).",
            weight_step="Resample each existing edge weight independently from N(0,1).",
            sign_step="Signs are unconstrained because draws are centered Gaussian.",
            formula="W'_ij ~ N(0,1) for (i,j) in Omega, else 0",
            note="Isolates effect of weight distribution at fixed topology.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                gaussian_weights=1,
            ),
        ),
        ArchitectureSpec(
            slug="06_er_matched_gaussian",
            title="CE-matched Erdos-Renyi with Gaussian weights",
            short_label="ER + Gaussian",
            topology_step="Generate ER(N=299,p=0.1), then trim edges to CE edge count.",
            weight_step="Assign N(0,1) weights on remaining edges.",
            sign_step="Signs are unconstrained.",
            formula="A ~ ER(299,0.1), nnz matched to CE; W_ij ~ N(0,1)",
            note="Random-graph baseline with CE-scale edge count.",
            props=_props(
                random_graph_model=1,
                rewire_connections=1,
                edge_count_matched_ce=1,
                gaussian_weights=1,
            ),
        ),
        ArchitectureSpec(
            slug="07_ws_matched_gaussian",
            title="CE-matched Watts-Strogatz with Gaussian weights",
            short_label="WS + Gaussian",
            topology_step="Generate WS(N=299,k=10,p=0.1) small-world graph.",
            weight_step="Assign N(0,1) weights on WS edges.",
            sign_step="Signs are unconstrained.",
            formula="A ~ WS(299,10,0.1); W_ij ~ N(0,1)",
            note="WS edge count is 2990, close to CE 3108.",
            props=_props(
                random_graph_model=1,
                rewire_connections=1,
                gaussian_weights=1,
            ),
        ),
        ArchitectureSpec(
            slug="08_ce_gaussian_abs_with_original_signs",
            title="CE architecture + |Gaussian| magnitudes with original signs",
            short_label="CE sign + |Gaussian|",
            topology_step="Keep CE adjacency fixed.",
            weight_step="Resample magnitudes from |N(0,1)|.",
            sign_step="Multiply by original CE sign on each edge.",
            formula="W'_ij = sign(W_ij) * |N(0,1)|",
            note="Preserves neurotransmitter sign at each connection.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                gaussian_weights=1,
                local_sign_preserved=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="09_ce_uniform_with_original_signs",
            title="CE architecture + Uniform magnitudes with original signs",
            short_label="CE sign + Uniform",
            topology_step="Keep CE adjacency fixed.",
            weight_step="Resample magnitudes from U(0,1).",
            sign_step="Multiply by original CE sign on each edge.",
            formula="W'_ij = sign(W_ij) * U(0,1)",
            note="Separates sign pattern from magnitude distribution.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                uniform_weights=1,
                local_sign_preserved=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="10_ce_sampled_with_original_signs",
            title="CE architecture + sampled CE magnitudes with original signs",
            short_label="CE sign + CE-sampled",
            topology_step="Keep CE adjacency fixed.",
            weight_step="Sample magnitudes from empirical CE positive/negative pools.",
            sign_step="Apply sign according to original CE edge sign.",
            formula="W'_ij = X^+ if W_ij>0 else X^- if W_ij<0",
            note="Samples from empirical CE weight distribution with sign conditioning.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                ce_sampled_weights=1,
                local_sign_preserved=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="11_ce_binary_with_original_signs",
            title="CE architecture + binary +/-1 with original signs",
            short_label="CE sign + binary",
            topology_step="Keep CE adjacency fixed.",
            weight_step="Set all nonzero magnitudes to 1.",
            sign_step="Use original CE sign at each edge (+1 or -1).",
            formula="W'_ij = sign(W_ij) for nonzero CE edges",
            note="Pure sign-only model with fixed unit magnitude.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                binary_weights=1,
                local_sign_preserved=1,
                global_sign_balance_preserved=1,
            ),
        ),
        ArchitectureSpec(
            slug="12_ce_binary_sign_balance_shuffled",
            title="CE architecture + binary +/-1 with shuffled signs (balance kept)",
            short_label="CE binary + sign shuffle",
            topology_step="Keep CE adjacency fixed.",
            weight_step="Set all nonzero magnitudes to 1.",
            sign_step="Randomly assign exactly num_neg edges as -1; rest +1.",
            formula="|W|=1 on support, with CE-matched global negative count",
            note="Preserves only global sign ratio, not per-edge sign identity.",
            props=_props(
                ce_topology=1,
                degree_preserved=1,
                edge_count_matched_ce=1,
                binary_weights=1,
                global_sign_balance_preserved=1,
            ),
        ),
    ]


def _draw_stage(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#2e2e2e",
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(x + w / 2, y + h * 0.38, fill(body, 28), ha="center", va="center", fontsize=9)


def make_variant_card(spec: ArchitectureSpec, out_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.2], wspace=0.22)

    ax_flow = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")

    y, h, w = 0.22, 0.56, 0.2
    xs = [0.02, 0.28, 0.54, 0.8]

    _draw_stage(ax_flow, xs[0], y, w, h, "Start", "Base matrix specification", "#e8f1ff")
    _draw_stage(ax_flow, xs[1], y, w, h, "Topology", spec.topology_step, "#eaf7ec")
    _draw_stage(ax_flow, xs[2], y, w, h, "Weights", spec.weight_step, "#fff3df")
    _draw_stage(ax_flow, xs[3], y, w, h, "Signs", spec.sign_step, "#fdecef")

    for i in range(3):
        x0 = xs[i] + w + 0.01
        x1 = xs[i + 1] - 0.01
        ax_flow.annotate(
            "",
            xy=(x1, y + h / 2),
            xytext=(x0, y + h / 2),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#4b4b4b"),
        )

    ax_flow.text(0.01, 0.04, f"Formula: {spec.formula}", fontsize=10, ha="left", va="bottom")
    ax_flow.text(0.01, -0.03, f"Note: {spec.note}", fontsize=9, ha="left", va="bottom")

    metrics = [
        ("Conn layout changed", int(spec.props["rewire_connections"] or spec.props["random_graph_model"])),
        ("Weight values changed", int(not spec.props["weights_preserved"])),
        ("Weight positions shuffled", spec.props["weight_shuffle"]),
        ("Local sign preserved", spec.props["local_sign_preserved"]),
        ("Global sign balance kept", spec.props["global_sign_balance_preserved"]),
        ("Random graph family", spec.props["random_graph_model"]),
    ]

    labels = [m[0] for m in metrics]
    values = np.array([m[1] for m in metrics], dtype=float)
    yloc = np.arange(len(labels))
    colors = ["#2a9d8f" if v > 0 else "#d9d9d9" for v in values]

    ax_bar.barh(yloc, values, color=colors, edgecolor="#5f5f5f")
    ax_bar.set_yticks(yloc, labels)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xticks([0, 1], labels=["No", "Yes"])
    ax_bar.invert_yaxis()
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.set_title("Variant fingerprint")

    for yi, val in enumerate(values):
        ax_bar.text(val + 0.03 if val > 0 else 0.03, yi, "Yes" if val > 0 else "No", va="center", fontsize=9)

    fig.suptitle(spec.title, fontsize=15, weight="bold", y=0.98)
    fig.savefig(out_dir / f"{spec.slug}.png", dpi=250)
    plt.close(fig)


def make_global_matrix(specs: list[ArchitectureSpec], out_dir: Path) -> None:
    row_labels = [s.short_label for s in specs]
    col_labels = [label for _, label in PROPERTY_ORDER]
    mat = np.array([[s.props[k] for k, _ in PROPERTY_ORDER] for s in specs], dtype=int)

    fig_h = max(8, 0.55 * len(row_labels) + 3)
    fig, ax = plt.subplots(figsize=(18, fig_h))

    cmap = "winter"
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title("Architecture map across topology, weight, and sign choices", fontsize=16, pad=16)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = "Y" if mat[i, j] == 1 else ""
            ax.text(j, i, txt, ha="center", va="center", color="white" if mat[i, j] == 1 else "black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_ticks([0, 1], labels=["No", "Yes"])
    cbar.set_label("Property present", rotation=90)

    fig.tight_layout()
    fig.savefig(out_dir / "00_architecture_map_matrix.png", dpi=280)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate architecture variant figures for thesis methods/results.")
    p.add_argument(
        "--outdir",
        default="architecture_variant_figures/output",
        help="Directory where PNG figures are written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_architectures()
    make_global_matrix(specs, out_dir)
    for spec in specs:
        make_variant_card(spec, out_dir)

    print(f"Wrote {1 + len(specs)} figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
