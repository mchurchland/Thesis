#!/usr/bin/env python3
"""
Delay-sensitivity validation for IPC with low polynomial orders (default K={1,2,3,4,5}).

This script helps justify D_max by checking:
1) capture of IPC(K, D_max) versus a reference delay D_ref
2) stability of model ranking versus D_ref (Pearson/Spearman on model means)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inv_arc_test import _pick_device
from network_stats.run_one import run_reservoir_with_pre
from util.util import (
    assign_random_unknown_signs,
    load_connectome,
    load_unknown_sign_weights,
    set_seed,
)

# Reuse existing sweep helpers to keep model construction and IPC logic identical.
from Parameter_justification.ipc_order_sweep_ce import (
    _build_model_reservoir,
    _expand_models,
    _model_title,
    _resolve_seeds,
    ipc_contrib_by_order,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IPC delay sensitivity for odd polynomial orders.")
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy")
    p.add_argument("--ce-unknown-sign-weights", default=None)
    p.add_argument(
        "--unknown-sign-policy",
        choices=("drop", "random_unknown_4to1"),
        default="drop",
    )
    p.add_argument("--unknown-sign-inhibitory-frac", type=float, default=0.2)
    p.add_argument("--unknown-sign-seed-offset", type=int, default=23_000_000)
    p.add_argument("--out-dir", default="ipc_delay_sensitivity")

    p.add_argument("--models", nargs="+", default=["all"], help="Model list or 'all'.")
    p.add_argument("--er-p", type=float, default=0.1)
    p.add_argument("--ws-p", type=float, default=0.1)

    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)

    p.add_argument("--rho", type=float, nargs="+", default=[0.95])
    p.add_argument("--leak", type=float, nargs="+", default=[0.8])
    p.add_argument("--input-scale", type=float, nargs="+", default=[1.0])

    p.add_argument("--washout", type=int, default=500)
    p.add_argument("--t-train", type=int, default=1500)
    p.add_argument("--t-test", type=int, default=500)
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument("--cuda", type=int, default=None)

    p.add_argument("--k-orders", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Order set K.")
    p.add_argument("--d-max-list", type=int, nargs="+", default=[30, 50, 80])
    p.add_argument("--d-ref", type=int, default=None, help="Reference delay. Default=max(d-max-list).")
    return p.parse_args()


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    d_values = sorted(set(int(d) for d in args.d_max_list if int(d) > 0))
    if not d_values:
        raise ValueError("--d-max-list must contain at least one positive value.")
    d_ref = args.d_ref if args.d_ref is not None else max(d_values)
    if d_ref not in d_values:
        d_values.append(d_ref)
        d_values = sorted(set(d_values))

    k_orders = sorted(set(int(k) for k in args.k_orders if int(k) >= 1))
    if not k_orders:
        raise ValueError("--k-orders must contain at least one order >= 1.")
    max_k = max(k_orders)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _pick_device(args.cuda)

    models = _expand_models(list(args.models))
    seeds = _resolve_seeds(args)

    ce_W_bio, _, _ = load_connectome(args.ce_adj, args.ce_ei)
    if ce_W_bio is None:
        raise FileNotFoundError(f"Could not load CE adjacency from: {args.ce_adj}")
    if not (0.0 <= args.unknown_sign_inhibitory_frac <= 1.0):
        raise ValueError("--unknown-sign-inhibitory-frac must be between 0 and 1.")
    ce_unknown_sign_weights = None
    if args.unknown_sign_policy == "random_unknown_4to1":
        ce_unknown_sign_weights = load_unknown_sign_weights(
            args.ce_adj,
            args.ce_unknown_sign_weights,
            n_nodes=ce_W_bio.shape[0],
        )
        if ce_unknown_sign_weights is None:
            raise FileNotFoundError(
                "random_unknown_4to1 requires an unknown-sign weight matrix. "
                "Regenerate the new connectome with util/read_xls.py or pass "
                "--ce-unknown-sign-weights explicitly."
            )

    raw_rows: list[tuple] = []
    # (model, seed, rho, leak, input_scale, d_max, ipc_kset)
    for seed in seeds:
        set_seed(seed)
        ce_W_trial = ce_W_bio
        if args.unknown_sign_policy == "random_unknown_4to1":
            ce_W_trial = assign_random_unknown_signs(
                ce_W_bio,
                ce_unknown_sign_weights,
                np.random.default_rng(seed + args.unknown_sign_seed_offset),
                inhibitory_fraction=args.unknown_sign_inhibitory_frac,
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

                        t_total = args.washout + args.t_train + args.t_test
                        u = (torch.rand(t_total, 1, device=device) * 2.0 - 1.0)
                        
                        X, _ = run_reservoir_with_pre(Wt, Win, u, leak)
                        Xtr = X[args.washout:args.washout + args.t_train]
                        Xte = X[args.washout + args.t_train:]
                        utr = u[args.washout:args.washout + args.t_train]
                        ute = u[args.washout + args.t_train:]

                        for d in d_values:
                            contrib = ipc_contrib_by_order(
                                Xtr=Xtr,
                                Xte=Xte,
                                utr=utr,
                                ute=ute,
                                max_delay=d,
                                max_order=max_k,
                                alpha=args.ridge_alpha,
                                device=device,
                            )
                            ipc_k = float(np.sum([contrib[k - 1] for k in k_orders if k <= len(contrib)]))
                            raw_rows.append((model, seed, rho, leak, input_scale, d, ipc_k))

    # Compute capture vs D_ref
    ref_map: dict[tuple, float] = {}
    for row in raw_rows:
        model, seed, rho, leak, in_scale, d, ipc_k = row
        if int(d) == int(d_ref):
            ref_map[(model, seed, rho, leak, in_scale)] = float(ipc_k)

    capture_rows: list[tuple] = []
    # (model, seed, rho, leak, input_scale, d_max, ipc_kset, capture_pct_vs_ref)
    for row in raw_rows:
        model, seed, rho, leak, in_scale, d, ipc_k = row
        ref = ref_map.get((model, seed, rho, leak, in_scale), np.nan)
        cap = (100.0 * float(ipc_k) / ref) if np.isfinite(ref) and ref > 0 else np.nan
        capture_rows.append((model, seed, rho, leak, in_scale, d, float(ipc_k), cap))

    # Summary by (model, d_max)
    summary_rows: list[tuple] = []
    for model in models:
        for d in d_values:
            vals = np.array(
                [r[6] for r in capture_rows if str(r[0]) == model and int(r[5]) == int(d)],
                dtype=np.float64,
            )
            caps = np.array(
                [r[7] for r in capture_rows if str(r[0]) == model and int(r[5]) == int(d)],
                dtype=np.float64,
            )
            summary_rows.append(
                (
                    model,
                    d,
                    float(np.mean(vals)) if vals.size else np.nan,
                    float(np.std(vals)) if vals.size else np.nan,
                    float(np.mean(caps)) if caps.size else np.nan,
                    float(np.std(caps)) if caps.size else np.nan,
                    int(vals.size),
                )
            )

    # Ranking stability of model means vs D_ref
    model_means_by_d: dict[int, dict[str, float]] = {}
    for d in d_values:
        mm: dict[str, float] = {}
        for model in models:
            vals = np.array(
                [r[6] for r in capture_rows if str(r[0]) == model and int(r[5]) == int(d)],
                dtype=np.float64,
            )
            mm[model] = float(np.mean(vals)) if vals.size else np.nan
        model_means_by_d[int(d)] = mm

    rank_rows: list[tuple] = []
    ref_models = [m for m in models if np.isfinite(model_means_by_d[int(d_ref)].get(m, np.nan))]
    ref_vec = np.array([model_means_by_d[int(d_ref)][m] for m in ref_models], dtype=np.float64)
    for d in d_values:
        cur_models = [m for m in ref_models if np.isfinite(model_means_by_d[int(d)].get(m, np.nan))]
        if len(cur_models) < 2:
            rank_rows.append((d, np.nan, np.nan, len(cur_models)))
            continue
        ref = np.array([model_means_by_d[int(d_ref)][m] for m in cur_models], dtype=np.float64)
        cur = np.array([model_means_by_d[int(d)][m] for m in cur_models], dtype=np.float64)
        pearson = float(np.corrcoef(ref, cur)[0, 1])
        spearman = float(np.corrcoef(_rankdata(ref), _rankdata(cur))[0, 1])
        rank_rows.append((d, pearson, spearman, len(cur_models)))

    raw_csv = out_dir / "ipc_delay_sensitivity_raw.csv"
    summary_csv = out_dir / "ipc_delay_sensitivity_summary.csv"
    rank_csv = out_dir / "ipc_delay_rank_stability.csv"
    report_txt = out_dir / "ipc_delay_sensitivity_report.txt"

    _write_csv(
        raw_csv,
        ["model", "seed", "rho_target", "leak", "input_scale", "d_max", "ipc_kset", "capture_pct_vs_ref"],
        capture_rows,
    )
    _write_csv(
        summary_csv,
        ["model", "d_max", "mean_ipc_kset", "std_ipc_kset", "mean_capture_pct", "std_capture_pct", "n_runs"],
        summary_rows,
    )
    _write_csv(
        rank_csv,
        ["d_max", "pearson_vs_ref", "spearman_vs_ref", "n_models"],
        rank_rows,
    )

    # Compact report
    lines: list[str] = []
    lines.append("IPC delay sensitivity report")
    lines.append(f"K={k_orders}, D candidates={d_values}, D_ref={d_ref}")
    lines.append("")
    caps_all = np.array([r[7] for r in capture_rows if np.isfinite(r[7])], dtype=np.float64)
    if caps_all.size:
        lines.append("[Global capture vs D_ref]")
        lines.append(
            f"capture% mean={float(np.mean(caps_all)):.2f}, std={float(np.std(caps_all)):.2f}, "
            f"min={float(np.min(caps_all)):.2f}, p05={float(np.percentile(caps_all, 5)):.2f}"
        )
    lines.append("")
    lines.append("[Ranking stability vs D_ref]")
    for d, pearson, spearman, n_models in rank_rows:
        lines.append(
            f"D_max={d}: pearson={pearson:.4f}, spearman={spearman:.4f}, n_models={n_models}"
        )
    lines.append("")
    lines.append("[Per-model means by D_max]")
    for model in models:
        model_name = _model_title(model, args.er_p)
        rows_m = [r for r in summary_rows if str(r[0]) == model]
        rows_m = sorted(rows_m, key=lambda x: int(x[1]))
        stats_txt = ", ".join(
            [f"D={int(r[1])}: {float(r[4]):.2f}%±{float(r[5]):.2f}%" for r in rows_m]
        )
        lines.append(f"{model_name}: {stats_txt}")
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {raw_csv}")
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {rank_csv}")
    print(f"[done] wrote {report_txt}")


if __name__ == "__main__":
    main()
