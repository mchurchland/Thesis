#!/usr/bin/env python3
"""
Sensitivity test for temporal window choices: washout, t_train, t_test.

Default behavior uses one-at-a-time scaling around baseline (e.g., 0.75x, 1.0x, 1.25x).
Outputs include capture-vs-baseline and TOST significance-pattern stability vs baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import warnings
from pathlib import Path

import numpy as np
import pingouin as pg
import torch

from inv_arc_test import SWEEP_LEAK, SWEEP_SR, SWEEP_U, _pick_device
from network_stats.run_one import run_reservoir_with_pre
from network_stats.stats import compute_IPC, compute_MC
from util.util import load_connectome

from ipc_order_sweep_ce import (
    _build_model_reservoir,
    _expand_models,
    _model_title,
    _resolve_seeds,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal-window sensitivity test.")
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy")
    p.add_argument("--out-dir", default="time_window_sensitivity")

    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--er-p", type=float, default=0.1)
    p.add_argument("--ws-p", type=float, default=0.1)

    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--seed-start", type=int, default=0)

    p.add_argument("--rho", type=float, nargs="+", default=list(SWEEP_SR))
    p.add_argument("--leak", type=float, nargs="+", default=list(SWEEP_LEAK))
    p.add_argument("--input-scale", type=float, nargs="+", default=list(SWEEP_U))

    p.add_argument("--washout", type=int, default=500)
    p.add_argument("--t-train", type=int, default=1500)
    p.add_argument("--t-test", type=int, default=500)
    p.add_argument("--ridge-alpha", type=float, default=1e-4)
    p.add_argument("--cuda", type=int, default=None)

    p.add_argument("--metric", choices=["ipc", "mc"], default="mc")
    p.add_argument("--k-orders", type=int, nargs="+", default=[1, 3, 5], help="Used for metric=ipc.")
    p.add_argument("--max-delay", type=int, default=30, help="Used for both ipc and mc.")

    p.add_argument("--scale-factors", type=float, nargs="+", default=[0.25, 0.75, 1.0, 1.25])
    p.add_argument("--full-grid", action="store_true", help="Use full factor grid instead of one-at-a-time.")
    p.add_argument(
        "--parallel-backend",
        choices=["none", "ray"],
        default="none",
        help="Execution backend. Use 'ray' for parallel task execution.",
    )
    p.add_argument("--ray-address", default=None, help="Optional Ray cluster address (e.g., 'auto').")
    p.add_argument("--ray-num-cpus", type=int, default=None, help="CPUs reserved for ray.init().")
    p.add_argument("--ray-num-gpus", type=float, default=None, help="GPUs reserved for ray.init().")
    p.add_argument("--ray-cpus-per-task", type=float, default=1.0, help="CPU resources per Ray task.")
    p.add_argument("--ray-gpus-per-task", type=float, default=0.0, help="GPU resources per Ray task.")
    return p.parse_args()


def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _scaled_int(v: int, factor: float, min_v: int) -> int:
    return max(min_v, int(round(v * factor)))


def _build_window_configs(
    washout: int,
    t_train: int,
    t_test: int,
    factors: list[float],
    full_grid: bool,
    min_len: int,
) -> list[tuple[str, int, int, int]]:
    facs = sorted(set(factors))
    configs: list[tuple[str, int, int, int]] = []

    if full_grid:
        for fw in facs:
            for ftr in facs:
                for fte in facs:
                    w = _scaled_int(washout, fw, min_len)
                    tr = _scaled_int(t_train, ftr, min_len)
                    te = _scaled_int(t_test, fte, min_len)
                    label = f"w{fw:.2f}_tr{ftr:.2f}_te{fte:.2f}"
                    configs.append((label, w, tr, te))
    else:
        configs.append(("baseline", washout, t_train, t_test))
        for f in facs:
            if np.isclose(f, 1.0):
                continue
            configs.append((f"washout_x{f:.2f}", _scaled_int(washout, f, min_len), t_train, t_test))
            configs.append((f"t_train_x{f:.2f}", washout, _scaled_int(t_train, f, min_len), t_test))
            configs.append((f"t_test_x{f:.2f}", washout, t_train, _scaled_int(t_test, f, min_len)))

    # keep order, dedupe
    seen = set()
    uniq: list[tuple[str, int, int, int]] = []
    for c in configs:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _stable_u_seed(
    model: str,
    seed: int,
    rho: float,
    leak: float,
    input_scale: float,
) -> int:
    key = f"{model}|{seed}|{rho:.8f}|{leak:.8f}|{input_scale:.8f}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    # Keep in signed 32-bit positive range for torch generator compatibility.
    return int.from_bytes(digest, "little") % (2**31 - 1)


def _evaluate_one_run(
    cfg_name: str,
    model: str,
    seed: int,
    rho: float,
    leak: float,
    input_scale: float,
    washout: int,
    t_train: int,
    t_test: int,
    metric: str,
    max_delay: int,
    ridge_alpha: float,
    k_orders: list[int],
    ce_W_bio: np.ndarray,
    nnz_target_ce: int,
    er_p: float,
    ws_p: float,
    device_str: str,
) -> tuple:
    device = torch.device(device_str)
    Wt, Win = _build_model_reservoir(
        model=model,
        ce_W_bio=ce_W_bio,
        seed=seed,
        target_sr=rho,
        input_scale=input_scale,
        device=device,
        er_p=er_p,
        ws_p=ws_p,
        nnz_target_ce=nnz_target_ce,
    )

    t_total = washout + t_train + t_test
    u_gen = torch.Generator(device="cpu")
    u_gen.manual_seed(_stable_u_seed(model, seed, rho, leak, input_scale))
    u = (torch.rand(t_total, 1, generator=u_gen) * 2.0 - 1.0).to(device)
    u = u - u.mean()

    X, _ = run_reservoir_with_pre(Wt, Win, u, leak)
    Xtr = X[washout:washout + t_train]
    Xte = X[washout + t_train:]
    utr = u[washout:washout + t_train]
    ute = u[washout + t_train:]

    if metric == "ipc":
        score = float(
            compute_IPC(
                Xtr=Xtr,
                Xte=Xte,
                utr=utr,
                ute=ute,
                max_delay=max_delay,
                alpha=ridge_alpha,
                device=device,
                orders=k_orders,
            )
        )
    else:
        score, _ = compute_MC(
            Xtr=Xtr,
            Xte=Xte,
            utr=utr,
            ute=ute,
            max_delay=max_delay,
            alpha=ridge_alpha,
            device=device,
        )
        score = float(score)

    return (
        cfg_name,
        model,
        seed,
        rho,
        leak,
        input_scale,
        washout,
        t_train,
        t_test,
        score,
    )


def main() -> None:
    args = parse_args()
    models = _expand_models(list(args.models))
    seeds = _resolve_seeds(args)
    if args.parallel_backend == "ray" and args.cuda is None:
        # Ray defaults to CPU execution unless user explicitly selects a CUDA index.
        device = torch.device("cpu")
    else:
        device = _pick_device(args.cuda)
    device_str = str(device)

    ce_W_bio, _, _ = load_connectome(args.ce_adj, args.ce_ei)
    if ce_W_bio is None:
        raise FileNotFoundError(f"Could not load CE adjacency from: {args.ce_adj}")
    nnz_target_ce = int((np.abs(ce_W_bio) > 0).sum())

    if args.metric == "ipc":
        k_orders = sorted(set(int(k) for k in args.k_orders if int(k) >= 1))
        if not k_orders:
            raise ValueError("--k-orders must contain at least one order >=1 for metric=ipc.")
    else:
        k_orders = []

    min_len = max(5, args.max_delay + 2)
    configs = _build_window_configs(
        washout=args.washout,
        t_train=args.t_train,
        t_test=args.t_test,
        factors=list(args.scale_factors),
        full_grid=bool(args.full_grid),
        min_len=min_len,
    )
    config_names = [c[0] for c in configs]
    if "baseline" not in config_names:
        configs = [("baseline", args.washout, args.t_train, args.t_test)] + configs
        config_names = [c[0] for c in configs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_specs: list[tuple] = []
    # (config, model, seed, rho, leak, input_scale, washout, t_train, t_test)
    for cfg_name, washout, t_train, t_test in configs:
        for seed in seeds:
            for rho in args.rho:
                for leak in args.leak:
                    for input_scale in args.input_scale:
                        for model in models:
                            run_specs.append(
                                (cfg_name, model, seed, rho, leak, input_scale, washout, t_train, t_test)
                            )

    raw_rows: list[tuple]
    if args.parallel_backend == "ray":
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError(
                "Ray backend requested, but 'ray' is not installed. "
                "Install it or rerun with --parallel-backend none."
            ) from exc

        init_kwargs: dict[str, object] = {"ignore_reinit_error": True}
        if args.ray_address is not None:
            init_kwargs["address"] = args.ray_address
        if args.ray_num_cpus is not None:
            init_kwargs["num_cpus"] = args.ray_num_cpus
        if args.ray_num_gpus is not None:
            init_kwargs["num_gpus"] = args.ray_num_gpus
        try:
            ray.init(**init_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Ray failed to initialize in this environment. "
                "Try --ray-address auto for an existing cluster or use --parallel-backend none."
            ) from exc

        try:
            remote_eval = ray.remote(_evaluate_one_run).options(
                num_cpus=float(args.ray_cpus_per_task),
                num_gpus=float(args.ray_gpus_per_task),
            )
            ce_ref = ray.put(ce_W_bio)
            futures = [
                remote_eval.remote(
                    cfg_name=cfg_name,
                    model=model,
                    seed=seed,
                    rho=rho,
                    leak=leak,
                    input_scale=input_scale,
                    washout=washout,
                    t_train=t_train,
                    t_test=t_test,
                    metric=args.metric,
                    max_delay=args.max_delay,
                    ridge_alpha=args.ridge_alpha,
                    k_orders=k_orders,
                    ce_W_bio=ce_ref,
                    nnz_target_ce=nnz_target_ce,
                    er_p=args.er_p,
                    ws_p=args.ws_p,
                    device_str=device_str,
                )
                for (cfg_name, model, seed, rho, leak, input_scale, washout, t_train, t_test) in run_specs
            ]
            raw_rows = ray.get(futures)
        finally:
            ray.shutdown()
    else:
        raw_rows = [
            _evaluate_one_run(
                cfg_name=cfg_name,
                model=model,
                seed=seed,
                rho=rho,
                leak=leak,
                input_scale=input_scale,
                washout=washout,
                t_train=t_train,
                t_test=t_test,
                metric=args.metric,
                max_delay=args.max_delay,
                ridge_alpha=args.ridge_alpha,
                k_orders=k_orders,
                ce_W_bio=ce_W_bio,
                nnz_target_ce=nnz_target_ce,
                er_p=args.er_p,
                ws_p=args.ws_p,
                device_str=device_str,
            )
            for (cfg_name, model, seed, rho, leak, input_scale, washout, t_train, t_test) in run_specs
        ]

    # Capture vs baseline (matched by run identity)
    ref_map: dict[tuple, float] = {}
    for r in raw_rows:
        cfg, model, seed, rho, leak, in_scale, *_rest, score = r
        if cfg == "baseline":
            ref_map[(model, seed, rho, leak, in_scale)] = float(score)

    capture_rows: list[tuple] = []
    # (...raw, capture_pct_vs_baseline)
    for r in raw_rows:
        cfg, model, seed, rho, leak, in_scale, washout, t_train, t_test, score = r
        ref = ref_map.get((model, seed, rho, leak, in_scale), np.nan)
        cap = (100.0 * float(score) / ref) if np.isfinite(ref) and ref > 0 else np.nan
        capture_rows.append(
            (cfg, model, seed, rho, leak, in_scale, washout, t_train, t_test, score, cap)
        )

    # Summary by (config, model)
    summary_rows: list[tuple] = []
    for cfg_name, washout, t_train, t_test in configs:
        for model in models:
            vals = np.array(
                [r[9] for r in capture_rows if str(r[0]) == cfg_name and str(r[1]) == model],
                dtype=np.float64,
            )
            caps = np.array(
                [r[10] for r in capture_rows if str(r[0]) == cfg_name and str(r[1]) == model],
                dtype=np.float64,
            )
            mean_v = float(np.mean(vals)) if vals.size else np.nan
            std_v = float(np.std(vals)) if vals.size else np.nan
            cv_v = (std_v / abs(mean_v)) if np.isfinite(mean_v) and abs(mean_v) > 1e-12 else np.nan
            summary_rows.append(
                (
                    cfg_name,
                    model,
                    washout,
                    t_train,
                    t_test,
                    mean_v,
                    std_v,
                    cv_v,
                    float(np.mean(caps)) if caps.size else np.nan,
                    float(np.std(caps)) if caps.size else np.nan,
                    int(vals.size),
                )
            )

    # TOST stability vs baseline:
    # compare model-pair significance patterns (p<0.05) and verify baseline-significant
    # pairs remain significant under each window config.
    score_by_cfg_model: dict[str, dict[str, np.ndarray]] = {}
    bound_by_cfg: dict[str, float] = {}
    for cfg_name, *_ in configs:
        mm: dict[str, np.ndarray] = {}
        vals_all: list[np.ndarray] = []
        for model in models:
            vals = np.array(
                [
                    r[9]
                    for r in capture_rows
                    if str(r[0]) == cfg_name and str(r[1]) == model and np.isfinite(float(r[9]))
                ],
                dtype=np.float64,
            )
            mm[model] = vals
            if vals.size:
                vals_all.append(vals)
        score_by_cfg_model[cfg_name] = mm
        if vals_all:
            all_cat = np.concatenate(vals_all)
            bound_by_cfg[cfg_name] = float(abs(np.median(all_cat)) * 0.05)
        else:
            bound_by_cfg[cfg_name] = np.nan

    pair_rows: list[tuple] = []
    pair_key_order = list(itertools.combinations(models, 2))
    for cfg_name, *_ in configs:
        bound = float(bound_by_cfg.get(cfg_name, np.nan))
        for model_a, model_b in pair_key_order:
            vals_a = score_by_cfg_model[cfg_name].get(model_a, np.array([], dtype=np.float64))
            vals_b = score_by_cfg_model[cfg_name].get(model_b, np.array([], dtype=np.float64))
            if (
                vals_a.size < 2
                or vals_b.size < 2
                or (not np.isfinite(bound))
                or bound <= 0.0
            ):
                pval = np.nan
                sig = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pingouin")
                    tost = pg.tost(vals_a, vals_b, bound, paired=False)
                pval = float(tost["pval"].iloc[0])
                sig = float(pval < 0.05) if np.isfinite(pval) else np.nan
            pair_rows.append(
                (
                    cfg_name,
                    model_a,
                    model_b,
                    bound,
                    pval,
                    sig,
                    int(vals_a.size),
                    int(vals_b.size),
                )
            )

    baseline_pair_sig: dict[tuple[str, str], float] = {}
    for cfg_name, model_a, model_b, _bound, _pval, sig, _na, _nb in pair_rows:
        if cfg_name == "baseline":
            baseline_pair_sig[(str(model_a), str(model_b))] = float(sig) if np.isfinite(sig) else np.nan

    rank_rows: list[tuple] = []
    baseline_bound = float(bound_by_cfg.get("baseline", np.nan))
    for cfg_name, *_ in configs:
        cur_pairs = [r for r in pair_rows if str(r[0]) == cfg_name]
        compared = 0
        same_sig = 0
        baseline_sig_n = 0
        baseline_sig_preserved = 0
        baseline_sig_lost = 0
        new_sig = 0
        for _cfg, model_a, model_b, _bound, _pval, sig, _na, _nb in cur_pairs:
            bs = baseline_pair_sig.get((str(model_a), str(model_b)), np.nan)
            if not (np.isfinite(bs) and np.isfinite(sig)):
                continue
            compared += 1
            if int(sig) == int(bs):
                same_sig += 1
            if int(bs) == 1:
                baseline_sig_n += 1
                if int(sig) == 1:
                    baseline_sig_preserved += 1
                else:
                    baseline_sig_lost += 1
            elif int(sig) == 1:
                new_sig += 1
        same_frac = (100.0 * float(same_sig) / float(compared)) if compared > 0 else np.nan
        bound = float(bound_by_cfg.get(cfg_name, np.nan))
        bound_delta = (bound - baseline_bound) if np.isfinite(bound) and np.isfinite(baseline_bound) else np.nan
        rank_rows.append(
            (
                cfg_name,
                bound,
                baseline_bound,
                bound_delta,
                compared,
                same_sig,
                same_frac,
                baseline_sig_n,
                baseline_sig_preserved,
                baseline_sig_lost,
                new_sig,
            )
        )

    raw_csv = out_dir / "time_window_sensitivity_raw.csv"
    summary_csv = out_dir / "time_window_sensitivity_summary.csv"
    rank_csv = out_dir / "time_window_rank_stability.csv"
    pair_csv = out_dir / "time_window_tost_pairs.csv"
    report_txt = out_dir / "time_window_sensitivity_report.txt"

    _write_csv(
        raw_csv,
        [
            "config",
            "model",
            "seed",
            "rho_target",
            "leak",
            "input_scale",
            "washout",
            "t_train",
            "t_test",
            "score",
            "capture_pct_vs_baseline",
        ],
        capture_rows,
    )
    _write_csv(
        summary_csv,
        [
            "config",
            "model",
            "washout",
            "t_train",
            "t_test",
            "mean_score",
            "std_score",
            "cv_score",
            "mean_capture_pct",
            "std_capture_pct",
            "n_runs",
        ],
        summary_rows,
    )
    _write_csv(
        rank_csv,
        [
            "config",
            "tost_bound",
            "baseline_tost_bound",
            "bound_delta_vs_baseline",
            "n_pairs_compared",
            "n_pairs_same_significance",
            "pct_pairs_same_significance",
            "n_baseline_significant",
            "n_baseline_significant_preserved",
            "n_baseline_significant_lost",
            "n_new_significant_vs_baseline",
        ],
        rank_rows,
    )
    _write_csv(
        pair_csv,
        [
            "config",
            "model_a",
            "model_b",
            "tost_bound",
            "tost_pval",
            "tost_significant",
            "n_model_a",
            "n_model_b",
        ],
        pair_rows,
    )

    lines: list[str] = []
    lines.append("Time-window sensitivity report")
    lines.append(f"metric={args.metric}, max_delay={args.max_delay}")
    lines.append(f"parallel_backend={args.parallel_backend}, device={device_str}, n_runs={len(run_specs)}")
    lines.append(f"rho_grid={list(args.rho)}")
    lines.append(f"leak_grid={list(args.leak)}")
    lines.append(f"input_scale_grid={list(args.input_scale)}")
    if args.metric == "ipc":
        lines.append(f"K={k_orders}")
    lines.append("")
    lines.append("[Global capture vs baseline]")
    for cfg_name, *_ in configs:
        caps = np.array([r[10] for r in capture_rows if str(r[0]) == cfg_name and np.isfinite(r[10])], dtype=np.float64)
        if caps.size == 0:
            continue
        lines.append(
            f"{cfg_name}: mean={float(np.mean(caps)):.2f}%, std={float(np.std(caps)):.2f}%, "
            f"min={float(np.min(caps)):.2f}%, p05={float(np.percentile(caps, 5)):.2f}%"
        )

    lines.append("")
    lines.append("[TOST stability vs baseline (model-pair significance)]")
    for (
        cfg_name,
        bound,
        baseline_bound,
        bound_delta,
        compared,
        same_sig,
        same_frac,
        baseline_sig_n,
        baseline_sig_preserved,
        baseline_sig_lost,
        new_sig,
    ) in rank_rows:
        preserved_txt = "YES" if int(baseline_sig_lost) == 0 else "NO"
        lines.append(
            f"{cfg_name}: bound={bound:.6g} (baseline={baseline_bound:.6g}, delta={bound_delta:.6g}), "
            f"same_sig={same_sig}/{compared} ({same_frac:.2f}%), "
            f"baseline_sig_preserved={baseline_sig_preserved}/{baseline_sig_n}, "
            f"baseline_sig_lost={baseline_sig_lost}, new_sig={new_sig}, "
            f"all_baseline_sig_preserved={preserved_txt}"
        )

    lines.append("")
    lines.append("[Per-model capture by config]")
    for model in models:
        model_name = _model_title(model, args.er_p)
        rows_m = [r for r in summary_rows if str(r[1]) == model]
        txt = ", ".join(
            [f"{str(r[0])}: {float(r[8]):.2f}%±{float(r[9]):.2f}%" for r in rows_m]
        )
        lines.append(f"{model_name}: {txt}")

    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {raw_csv}")
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {rank_csv}")
    print(f"[done] wrote {pair_csv}")
    print(f"[done] wrote {report_txt}")


if __name__ == "__main__":
    main()
