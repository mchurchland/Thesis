#!/usr/bin/env python3
"""
Test IPC degradation vs. Legendre polynomial degree on the C. elegans connectome.

This script reuses network_stats.stats.compute_IPC with shared helpers from
the existing codebase (legendre_P, set_seed, and _pick_device).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inv_arc_test import ALL_JOB_KEYS, _pick_device
from network_stats.run_one import run_reservoir_with_pre
import network_stats.stats as stats_mod
from reservoir_variants import VARIANT_LABELS
from util.util import (
    _conn_and_w_shuffle_ce,
    _conn_shuffle_ce,
    _sample_from_cel,
    _shuffle_ce_weights,
    assign_random_unknown_signs,
    build_reservoir,
    load_connectome,
    load_unknown_sign_weights,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep IPC by polynomial order on the C. elegans connectome."
    )
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy", help="Path to CE adjacency npy.")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy", help="Path to CE EI labels npy.")
    p.add_argument(
        "--ce-unknown-sign-weights",
        default=None,
        help="Optional .npy matrix with magnitudes for new-connectome complex/no-pred edges.",
    )
    p.add_argument(
        "--unknown-sign-policy",
        choices=("drop", "random_unknown_sign_matched"),
        default="drop",
        help="Use random_unknown_sign_matched to add complex/no-pred edges once per seed.",
    )
    p.add_argument(
        "--unknown-sign-seed-offset",
        type=int,
        default=23_000_000,
        help="Offset added to each seed before randomizing complex/no-pred signs.",
    )
    p.add_argument("--out-dir", default="ipc_order_sweep_ce", help="Directory for outputs.")
    p.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Reservoir models to compare. Use 'all' for inv_arc_test.ALL_JOB_KEYS.",
    )
    p.add_argument("--er-p", type=float, default=0.1, help="ER edge probability.")
    p.add_argument("--ws-p", type=float, default=0.1, help="WS rewiring probability.")

    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list.")
    p.add_argument("--n-seeds", type=int, default=20, help="Number of seeds (uses --seed-start).")
    p.add_argument("--seed-start", type=int, default=0, help="Starting seed for --n-seeds.")
    p.add_argument("--rho", type=float, nargs="+", default=[0.95], help="Spectral radius target(s).")
    p.add_argument("--leak", type=float, nargs="+", default=[0.8], help="Leak rate(s).")
    p.add_argument(
        "--input-scale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Input scale(s) for Win.",
    )

    p.add_argument("--washout", type=int, default=500)
    p.add_argument("--t-train", type=int, default=1500)
    p.add_argument("--t-test", type=int, default=500)
    p.add_argument("--max-delay", type=int, default=50)
    p.add_argument("--max-order", type=int, default=10)
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument(
        "--capacity-thresholds",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "Squared-correlation thresholds for retaining IPC terms. "
            "Default 0.0 preserves the old behavior."
        ),
    )
    p.add_argument(
        "--neuron-biases",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "Per-neuron random bias uniform half-ranges. "
            "Each trial seed samples b_i ~ Uniform(-value, value)."
        ),
    )
    p.add_argument("--cuda", type=int, default=None, help="CUDA index. Omit for auto.")
    return p.parse_args()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _expand_models(models_arg: list[str]) -> list[str]:
    if "all" not in models_arg:
        return _dedupe_keep_order(models_arg)
    expanded = list(ALL_JOB_KEYS)
    for m in models_arg:
        if m != "all":
            expanded.append(m)
    return _dedupe_keep_order(expanded)


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None and len(args.seeds) > 0:
        return list(args.seeds)
    if args.n_seeds is not None:
        if args.n_seeds < 1:
            raise ValueError("--n-seeds must be >= 1")
        return list(range(args.seed_start, args.seed_start + args.n_seeds))
    # default fallback keeps behavior close to previous script defaults
    return list(range(args.seed_start, args.seed_start + 5))


def _build_model_reservoir(
    model: str,
    ce_W_bio: np.ndarray,
    seed: int,
    target_sr: float,
    input_scale: float,
    device: torch.device,
    er_p: float,
    ws_p: float,
    nnz_target_ce: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    feature_conn = "cel"
    ce_source = ce_W_bio
    nnz_target: int | None = None

    if model in ("cel", "real"):
        feature_conn = "cel"
    elif model == "shuffle_weights":
        ce_source = _shuffle_ce_weights(ce_W_bio, rng)
        feature_conn = "cel"
    elif model == "conn_shuf":
        ce_source = _conn_and_w_shuffle_ce(ce_W_bio, rng)
        feature_conn = "cel"
    elif model == "conn_shuf_only":
        ce_source = _conn_shuffle_ce(ce_W_bio, rng)
        feature_conn = "cel"
    elif model == "cel_sample":
        ce_source = _sample_from_cel(ce_W_bio, rng)
        feature_conn = "cel"
    elif model in ("er", "er_randN"):
        feature_conn = f"er_p={er_p}"
        nnz_target = nnz_target_ce
    elif model == "ws_p01_randN":
        feature_conn = f"ws_p={ws_p}"
        nnz_target = nnz_target_ce
    elif model in (
        "cel_randN",
        "local_sign",
        "local_sign+flat",
        "local_sign+sample",
        "local_sign+binary",
        "global_sign_pres",
        "global_sign_pres_real_weight",
        "binary_base",
        "binary_base_topology_shuffle",
        "binary+conshuffle+wshuffle",
        "binary+shuffle",
    ):
        feature_conn = model
    else:
        raise ValueError(
            f"Model '{model}' is not supported in ipc_order_sweep_ce.py yet."
        )

    return build_reservoir(
        target_sr=target_sr,
        input_scale=input_scale,
        seed=seed,
        feature_conn=feature_conn,
        N=ce_W_bio.shape[0],
        ce_W_bio=ce_source,
        nnz_target=nnz_target,
        DEVICE=device,
    )


def _make_neuron_bias(
    n_nodes: int,
    bias_range: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor | None:
    if bias_range == 0.0:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.empty(n_nodes, device=device, dtype=dtype).uniform_(
        -float(bias_range),
        float(bias_range),
        generator=gen,
    )


def ipc_contrib_by_order(
    Xtr: Tensor,
    Xte: Tensor,
    utr: Tensor,
    ute: Tensor,
    max_delay: int,
    max_order: int,
    alpha: float,
    device: torch.device,
    capacity_threshold: float = 0.0,
) -> np.ndarray:
    scores = ipc_scores_by_order_delay(
        Xtr=Xtr,
        Xte=Xte,
        utr=utr,
        ute=ute,
        max_delay=max_delay,
        max_order=max_order,
        alpha=alpha,
        device=device,
    )
    return ipc_contrib_from_scores(scores, capacity_threshold)


def ipc_scores_by_order_delay(
    Xtr: Tensor,
    Xte: Tensor,
    utr: Tensor,
    ute: Tensor,
    max_delay: int,
    max_order: int,
    alpha: float,
    device: torch.device,
) -> np.ndarray:
    """Return corr^2 scores with shape [order, delay] before thresholding."""
    scores = np.zeros((max_order, max_delay), dtype=np.float64)
    orders = list(range(1, max_order + 1))

    for d in range(1, max_delay + 1):
        base_tr = utr[:-d]
        base_te = ute[:-d]
        Xtr_d = Xtr[d:]
        Xte_d = Xte[d:]

        ytr_cols = []
        yte_cols = []
        for k in orders:
            ytr_k = stats_mod.legendre_P(base_tr, k)
            yte_k = stats_mod.legendre_P(base_te, k)
            if ytr_k.dim() == 1:
                ytr_k = ytr_k.unsqueeze(1)
            if yte_k.dim() == 1:
                yte_k = yte_k.unsqueeze(1)
            ytr_cols.append(ytr_k)
            yte_cols.append(yte_k)

        Ytr = torch.cat(ytr_cols, dim=1).to(device)
        Yte = torch.cat(yte_cols, dim=1).to(device)
        Yhat = stats_mod.ridge_fit_predict(Xtr_d, Ytr, Xte_d, alpha, DEVICE=device)

        for j in range(Ytr.shape[1]):
            scores[j, d - 1] = stats_mod.corr2_score(Yte[:, j], Yhat[:, j])

    return scores


def ipc_contrib_from_scores(scores: np.ndarray, capacity_threshold: float = 0.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    kept = np.where(scores >= float(capacity_threshold), scores, 0.0)
    return kept.sum(axis=1)


def run_ipc_scores_once(
    Wt: Tensor,
    Win: Tensor,
    leak: float,
    washout: int,
    t_train: int,
    t_test: int,
    max_delay: int,
    max_order: int,
    ridge_alpha: float,
    device: torch.device,
    neuron_bias: float,
    bias_seed: int,
) -> np.ndarray:
    t_total = washout + t_train + t_test
    u = (torch.rand(t_total, 1, device=device) * 2.0 - 1.0) ## rescale to [-1, 1]

    bias_vec = _make_neuron_bias(
        Wt.shape[0],
        neuron_bias,
        device,
        Wt.dtype,
        seed=bias_seed,
    )
    X, _ = run_reservoir_with_pre(Wt, Win, u, leak, bias=bias_vec)
    Xtr = X[washout:washout + t_train]
    Xte = X[washout + t_train:]
    utr = u[washout:washout + t_train]
    ute = u[washout + t_train:]

    return ipc_scores_by_order_delay(
        Xtr=Xtr,
        Xte=Xte,
        utr=utr,
        ute=ute,
        max_delay=max_delay,
        max_order=max_order,
        alpha=ridge_alpha,
        device=device,
    )


def write_raw_csv(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "seed",
                "rho_target",
                "leak",
                "input_scale",
                "neuron_bias",
                "capacity_threshold",
                "order",
                "parity",
                "ipc_contrib",
            ]
        )
        w.writerows(rows)


def write_summary_csv(path: Path, stats_rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "neuron_bias",
                "capacity_threshold",
                "order",
                "parity",
                "mean_ipc_contrib",
                "std_ipc_contrib",
                "median_ipc_contrib",
                "ratio_to_order3",
            ]
        )
        w.writerows(stats_rows)


def write_validation_csv(path: Path, rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "seed",
                "rho_target",
                "leak",
                "input_scale",
                "neuron_bias",
                "capacity_threshold",
                "ipc_total",
                "ipc_odd",
                "ipc_even",
                "odd_share_pct",
                "even_share_pct",
                "ipc_12345",
                "capture_pct_12345",
            ]
        )
        w.writerows(rows)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _model_title(model: str, er_p: float) -> str:
    if "|bias=" in model and "|thr=" in model:
        base, rest = model.split("|bias=", 1)
        bias, threshold = rest.split("|thr=", 1)
        return f"{_model_title(base, er_p)} | bias={bias} | threshold={threshold}"
    if model in ("cel", "real"):
        return "C. elegans"
    if model in ("er", "er_randN"):
        return f"ER (p={er_p:g})"
    if model in VARIANT_LABELS:
        return VARIANT_LABELS[model]
    return model


def _report_block_for_model(
    summary: dict[int, dict[str, float]],
    first5_stats: dict[str, float] | None = None,
    odd_even_stats: dict[str, float] | None = None,
) -> list[str]:
    order3 = summary.get(3, {}).get("mean", np.nan)
    order4 = summary.get(4, {}).get("mean", np.nan)
    order5 = summary.get(5, {}).get("mean", np.nan)
    order1 = summary.get(1, {}).get("mean", np.nan)
    order2 = summary.get(2, {}).get("mean", np.nan)

    tail_orders = [k for k in summary.keys() if k > 3]
    tail_ratios = []
    for k in tail_orders:
        m = summary[k]["mean"]
        if np.isfinite(order3) and order3 > 0.0:
            tail_ratios.append(m / order3)

    finite_means = [v["mean"] for v in summary.values() if np.isfinite(v["mean"])]
    total_ipc_mean = float(np.sum(finite_means)) if finite_means else np.nan

    first3 = 0.0
    for k in (1, 2, 3):
        if k in summary and np.isfinite(summary[k]["mean"]):
            first3 += summary[k]["mean"]

    first5 = 0.0
    for k in (1, 2, 3, 4, 5):
        if k in summary and np.isfinite(summary[k]["mean"]):
            first5 += summary[k]["mean"]

    first3_pct = np.nan
    if np.isfinite(total_ipc_mean) and total_ipc_mean > 0.0:
        first3_pct = 100.0 * first3 / total_ipc_mean

    first5_pct = np.nan
    if np.isfinite(total_ipc_mean) and total_ipc_mean > 0.0:
        first5_pct = 100.0 * first5 / total_ipc_mean

    order1_pct = np.nan
    order2_pct = np.nan
    order3_pct = np.nan
    order5_pct = np.nan
    if np.isfinite(total_ipc_mean) and total_ipc_mean > 0.0:
        if np.isfinite(order1):
            order1_pct = 100.0 * order1 / total_ipc_mean
        if np.isfinite(order2):
            order2_pct = 100.0 * order2 / total_ipc_mean
        if np.isfinite(order3):
            order3_pct = 100.0 * order3 / total_ipc_mean
        if np.isfinite(order5):
            order5_pct = 100.0 * order5 / total_ipc_mean

    first_below_half = None
    if np.isfinite(order3) and order3 > 0.0:
        for k in sorted(tail_orders):
            if summary[k]["mean"] <= 0.5 * order3:
                first_below_half = k
                break

    o1_std = summary.get(1, {}).get("std", np.nan)
    o3_std = summary.get(3, {}).get("std", np.nan)
    o5_std = summary.get(5, {}).get("std", np.nan)
    o1_mean = summary.get(1, {}).get("mean", np.nan)
    o3_mean = summary.get(3, {}).get("mean", np.nan)
    o5_mean = summary.get(5, {}).get("mean", np.nan)

    lines: list[str] = []
    lines.append(f"IPC share in first 3 orders: {first3_pct:.2f}%")
    lines.append(
        f"Odd contributions: o1={o1_mean:.6f}, o3={o3_mean:.6f}, o5={o5_mean:.6f}"
    )
    lines.append(
        f"Odd contribution stds: o1={o1_std:.6f}, o3={o3_std:.6f}, o5={o5_std:.6f}"
    )

    if first5_stats is not None:
        if odd_even_stats is not None:
            lines.append(
                f"Odd-order contribution: {odd_even_stats['odd_mean']:.6f} ± {odd_even_stats['odd_std']:.6f} (std)"
            )
            lines.append(
                f"Even-order contribution: {odd_even_stats['even_mean']:.6f} ± {odd_even_stats['even_std']:.6f} (std)"
            )
            lines.append(
                f"Odd/even share: odd={odd_even_stats['odd_pct_mean']:.2f}% ± "
                f"{odd_even_stats['odd_pct_std']:.2f}%, even={odd_even_stats['even_pct_mean']:.2f}% ± "
                f"{odd_even_stats['even_pct_std']:.2f}%"
            )
        lines.append(
            f"Orders 1+2+3+4+5 contribution: {first5_stats['mean']:.6f} ± {first5_stats['std']:.6f} (std)"
        )
        lines.append(
            f"IPC share in orders 1+2+3+4+5: {first5_stats['pct_mean']:.2f}% ± {first5_stats['pct_std']:.2f}% (std)"
        )
    else:
        lines.append(f"IPC share in orders 1+2+3+4+5: {first5_pct:.2f}%")

    return lines


def write_report(
    path: Path,
    summaries_by_model: dict[str, dict[int, dict[str, float]]],
    first5_stats_by_model: dict[str, dict[str, float] | None],
    odd_even_stats_by_model: dict[str, dict[str, float] | None],
    models: list[str],
    er_p: float,
    validation_rows: list[tuple],
    max_order: int,
) -> None:
    lines: list[str] = []
    lines.append("IPC order-degradation report (model comparison)")
    for model in models:
        if model not in summaries_by_model:
            continue
        lines.append("")
        lines.append(f"[{_model_title(model, er_p)}]")
        lines.extend(
            _report_block_for_model(
                summaries_by_model[model],
                first5_stats=first5_stats_by_model.get(model),
                odd_even_stats=odd_even_stats_by_model.get(model),
            )
        )

    ce_key = "real" if "real" in summaries_by_model else ("cel" if "cel" in summaries_by_model else None)
    er_key = "er_randN" if "er_randN" in summaries_by_model else ("er" if "er" in summaries_by_model else None)
    if ce_key is not None and er_key is not None:
        ce_o3 = summaries_by_model[ce_key].get(3, {}).get("mean", np.nan)
        er_o3 = summaries_by_model[er_key].get(3, {}).get("mean", np.nan)
        if np.isfinite(ce_o3) and ce_o3 > 0 and np.isfinite(er_o3):
            lines.append("")
            lines.append("[CE vs ER at order 3]")
            lines.append(f"ER / CE order-3 mean ratio: {er_o3 / ce_o3:.6f}")

    if validation_rows:
        totals = np.array([float(r[7]) for r in validation_rows], dtype=np.float64)
        first5 = np.array([float(r[12]) for r in validation_rows], dtype=np.float64)
        caps = np.array([float(r[13]) for r in validation_rows], dtype=np.float64)
        valid = np.isfinite(totals) & np.isfinite(first5) & np.isfinite(caps)
        totals = totals[valid]
        first5 = first5[valid]
        caps = caps[valid]

        lines.append("")
        lines.append("[1,2,3,4,5 validation]")
        if caps.size == 0:
            lines.append("No finite nonzero IPC totals after thresholding.")
        else:
            pearson = np.nan
            spearman = np.nan
            if len(totals) >= 2:
                pearson = float(np.corrcoef(totals, first5)[0, 1])
                spearman = float(np.corrcoef(_rankdata(totals), _rankdata(first5))[0, 1])

            lines.append(
                f"capture% mean={float(np.mean(caps)):.2f}, std={float(np.std(caps)):.2f}, "
                f"min={float(np.min(caps)):.2f}, p05={float(np.percentile(caps, 5)):.2f}"
            )
            lines.append(
                f"agreement IPC(1..{max_order}) vs IPC(1,2,3,4,5): "
                f"pearson={pearson:.4f}, spearman={spearman:.4f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, stats_rows: list[tuple], er_p: float) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)
    color_map = {"cel": "#1f77b4", "er": "#ff7f0e"}
    keys = sorted({(str(r[0]), float(r[1]), float(r[2])) for r in stats_rows})
    for idx, (model, neuron_bias, threshold) in enumerate(keys):
        rows = [
            r for r in stats_rows
            if str(r[0]) == model and np.isclose(float(r[1]), neuron_bias) and np.isclose(float(r[2]), threshold)
        ]
        rows = sorted(rows, key=lambda x: int(x[3]))
        orders = np.array([int(r[3]) for r in rows], dtype=np.int32)
        means = np.array([float(r[5]) for r in rows], dtype=np.float64)
        stds = np.array([float(r[6]) for r in rows], dtype=np.float64)
        color = color_map.get(model, f"C{idx}")
        label = f"{_model_title(model, er_p)} | bias={neuron_bias:g} | thr={threshold:g}"
        ax.plot(orders, means, marker="o", linewidth=2.0, color=color, label=f"{label} mean")
        ax.fill_between(
            orders,
            np.maximum(0.0, means - stds),
            means + stds,
            color=color,
            alpha=0.15,
            label=f"{label} ±1 std",
        )

    ax.axvline(3, color="#d62728", linestyle="--", linewidth=1.5, label="Order 3 cutoff")
    ax.set_xlabel("Legendre polynomial order")
    ax.set_ylabel("IPC contribution")
    all_orders = sorted({int(r[3]) for r in stats_rows})
    ax.set_title("IPC contribution vs polynomial order")
    ax.set_xticks(all_orders)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_order < 1:
        raise ValueError("--max-order must be >= 1")
    if args.max_delay < 1:
        raise ValueError("--max-delay must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.cuda)
    models = _expand_models(list(args.models))
    seeds = _resolve_seeds(args)

    ce_W_bio, _, _ = load_connectome(args.ce_adj, args.ce_ei)
    if ce_W_bio is None:
        raise FileNotFoundError(f"Could not load CE adjacency from: {args.ce_adj}")
    ce_unknown_sign_weights = None
    if args.unknown_sign_policy == "random_unknown_sign_matched":
        ce_unknown_sign_weights = load_unknown_sign_weights(
            args.ce_adj,
            args.ce_unknown_sign_weights,
            n_nodes=ce_W_bio.shape[0],
        )
        if ce_unknown_sign_weights is None:
            raise FileNotFoundError(
                "random_unknown_sign_matched requires an unknown-sign weight matrix. "
                "Regenerate the new connectome with util/read_xls.py or pass "
                "--ce-unknown-sign-weights explicitly."
            )
    neuron_biases = [float(b) for b in args.neuron_biases]
    if any(b < 0.0 for b in neuron_biases):
        raise ValueError("--neuron-biases values must be non-negative.")
    capacity_thresholds = [float(t) for t in args.capacity_thresholds]
    if any(t < 0.0 for t in capacity_thresholds):
        raise ValueError("--capacity-thresholds values must be non-negative.")

    raw_rows: list[tuple] = []
    for seed in seeds:
        set_seed(seed)
        ce_W_trial = ce_W_bio
        if args.unknown_sign_policy == "random_unknown_sign_matched":
            ce_W_trial = assign_random_unknown_signs(
                ce_W_bio,
                ce_unknown_sign_weights,
                np.random.default_rng(seed + args.unknown_sign_seed_offset),
            )
        nnz_target_ce = int((np.abs(ce_W_trial) > 0).sum())
        for rho in args.rho:
            for leak in args.leak:
                for input_scale in args.input_scale:
                    for model in models:
                        Wt, Win = _build_model_reservoir(
                            model=model,
                            ce_W_bio=ce_W_trial,
                            seed=seed,
                            target_sr=rho,
                            input_scale=input_scale,
                            device=device,
                            er_p=args.er_p,
                            ws_p=args.ws_p,
                            nnz_target_ce=nnz_target_ce,
                        )

                        for neuron_bias in neuron_biases:
                            scores = run_ipc_scores_once(
                                Wt=Wt,
                                Win=Win,
                                leak=leak,
                                washout=args.washout,
                                t_train=args.t_train,
                                t_test=args.t_test,
                                max_delay=args.max_delay,
                                max_order=args.max_order,
                                ridge_alpha=args.ridge_alpha,
                                device=device,
                                neuron_bias=neuron_bias,
                                bias_seed=seed + 17_000_003,
                            )
                            for capacity_threshold in capacity_thresholds:
                                contrib = ipc_contrib_from_scores(
                                    scores,
                                    capacity_threshold=capacity_threshold,
                                )

                                for order in range(1, args.max_order + 1):
                                    raw_rows.append(
                                        (
                                            model,
                                            seed,
                                            rho,
                                            leak,
                                            input_scale,
                                            neuron_bias,
                                            capacity_threshold,
                                            order,
                                            "odd" if order % 2 else "even",
                                            float(contrib[order - 1]),
                                        )
                                    )

    summaries_by_model: dict[str, dict[int, dict[str, float]]] = {}
    first5_stats_by_model: dict[str, dict[str, float] | None] = {}
    odd_even_stats_by_model: dict[str, dict[str, float] | None] = {}
    summary_rows: list[tuple] = []
    validation_rows: list[tuple] = []

    condition_keys = sorted(
        {(str(r[0]), float(r[5]), float(r[6])) for r in raw_rows},
        key=lambda x: (x[0], x[1], x[2]),
    )

    for model, neuron_bias, capacity_threshold in condition_keys:
        condition_label = f"{model}|bias={neuron_bias:g}|thr={capacity_threshold:g}"
        rows_m = [
            r for r in raw_rows
            if str(r[0]) == model
            and np.isclose(float(r[5]), neuron_bias)
            and np.isclose(float(r[6]), capacity_threshold)
        ]
        by_order: dict[int, list[float]] = {k: [] for k in range(1, args.max_order + 1)}
        for row in rows_m:
            order = int(row[7])
            val = float(row[9])
            by_order[order].append(val)

        runs: dict[tuple[int, float, float, float, float, float], dict[int, float]] = {}
        for row in rows_m:
            key = (
                int(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
            )
            order = int(row[7])
            val = float(row[9])
            if key not in runs:
                runs[key] = {}
            runs[key][order] = val

        first5_vals: list[float] = []
        first5_pct_vals: list[float] = []
        odd_vals: list[float] = []
        even_vals: list[float] = []
        odd_pct_vals: list[float] = []
        even_pct_vals: list[float] = []
        for run_key, run_orders in runs.items():
            total = float(sum(run_orders.values()))
            odd = float(sum(v for k, v in run_orders.items() if k % 2 == 1))
            even = float(sum(v for k, v in run_orders.items() if k % 2 == 0))
            first5 = float(sum(run_orders.get(k, 0.0) for k in (1, 2, 3, 4, 5)))
            odd_vals.append(odd)
            even_vals.append(even)
            first5_vals.append(first5)
            odd_share = (100.0 * odd / total) if total > 0.0 else np.nan
            even_share = (100.0 * even / total) if total > 0.0 else np.nan
            cap = (100.0 * first5 / total) if total > 0.0 else np.nan
            if total > 0.0:
                odd_pct_vals.append(odd_share)
                even_pct_vals.append(even_share)
                first5_pct_vals.append(cap)
            seed_k, rho_k, leak_k, in_k, bias_k, threshold_k = run_key
            validation_rows.append(
                (
                    model,
                    seed_k,
                    rho_k,
                    leak_k,
                    in_k,
                    bias_k,
                    threshold_k,
                    total,
                    odd,
                    even,
                    odd_share,
                    even_share,
                    first5,
                    cap,
                )
            )

        first5_stats = None
        if first5_vals:
            first5_stats = {
                "mean": float(np.mean(np.array(first5_vals, dtype=np.float64))),
                "std": float(np.std(np.array(first5_vals, dtype=np.float64))),
                "pct_mean": float(np.mean(np.array(first5_pct_vals, dtype=np.float64))) if first5_pct_vals else np.nan,
                "pct_std": float(np.std(np.array(first5_pct_vals, dtype=np.float64))) if first5_pct_vals else np.nan,
            }
        first5_stats_by_model[condition_label] = first5_stats

        odd_even_stats = None
        if odd_vals or even_vals:
            odd_even_stats = {
                "odd_mean": float(np.mean(np.array(odd_vals, dtype=np.float64))) if odd_vals else np.nan,
                "odd_std": float(np.std(np.array(odd_vals, dtype=np.float64))) if odd_vals else np.nan,
                "even_mean": float(np.mean(np.array(even_vals, dtype=np.float64))) if even_vals else np.nan,
                "even_std": float(np.std(np.array(even_vals, dtype=np.float64))) if even_vals else np.nan,
                "odd_pct_mean": float(np.mean(np.array(odd_pct_vals, dtype=np.float64))) if odd_pct_vals else np.nan,
                "odd_pct_std": float(np.std(np.array(odd_pct_vals, dtype=np.float64))) if odd_pct_vals else np.nan,
                "even_pct_mean": float(np.mean(np.array(even_pct_vals, dtype=np.float64))) if even_pct_vals else np.nan,
                "even_pct_std": float(np.std(np.array(even_pct_vals, dtype=np.float64))) if even_pct_vals else np.nan,
            }
        odd_even_stats_by_model[condition_label] = odd_even_stats

        order3_mean = float(np.mean(by_order[3])) if 3 in by_order and by_order[3] else np.nan
        summary: dict[int, dict[str, float]] = {}
        for order in range(1, args.max_order + 1):
            vals = np.array(by_order[order], dtype=np.float64)
            mean_v = float(np.mean(vals)) if vals.size else np.nan
            std_v = float(np.std(vals)) if vals.size else np.nan
            med_v = float(np.median(vals)) if vals.size else np.nan
            ratio = float(mean_v / order3_mean) if np.isfinite(order3_mean) and order3_mean > 0 else np.nan
            summary[order] = {"mean": mean_v, "std": std_v, "median": med_v, "ratio_to_order3": ratio}
            summary_rows.append(
                (
                    model,
                    neuron_bias,
                    capacity_threshold,
                    order,
                    "odd" if order % 2 else "even",
                    mean_v,
                    std_v,
                    med_v,
                    ratio,
                )
            )
        summaries_by_model[condition_label] = summary

    report_models = list(summaries_by_model.keys())

    raw_csv = out_dir / "ipc_by_order_raw.csv"
    summary_csv = out_dir / "ipc_by_order_summary.csv"
    validation_csv = out_dir / "ipc_odd_even_validation.csv"
    report_txt = out_dir / "ipc_order_degradation_report.txt"
    plot_png = out_dir / "ipc_by_order_contrib.png"

    write_raw_csv(raw_csv, raw_rows)
    write_summary_csv(summary_csv, summary_rows)
    write_validation_csv(validation_csv, validation_rows)
    write_report(
        report_txt,
        summaries_by_model,
        first5_stats_by_model,
        odd_even_stats_by_model,
        models=report_models,
        er_p=args.er_p,
        validation_rows=validation_rows,
        max_order=args.max_order,
    )
    write_plot(plot_png, summary_rows, er_p=args.er_p)

    print(f"[done] wrote {raw_csv}")
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {validation_csv}")
    print(f"[done] wrote {report_txt}")
    print(f"[done] wrote {plot_png}")


if __name__ == "__main__":
    main()
