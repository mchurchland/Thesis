#!/usr/bin/env python3
"""Check whether GR architecture ordering changes with common-tail length.

This is a thin parameter-justification driver around the production GR code.
Reservoir construction, input generation, state evolution, effective-rank
calculation, model expansion, connectome loading, and seed handling all reuse
existing project helpers.

The independently sampled prefix remains fixed at seven steps by default;
only the duration of the shared continuation changes, so each condition has
total length ``prefix_length + common_tail_length``.

Absolute GR remains conditional on this shared-tail protocol. The report
therefore emphasizes architecture-order stability relative to the production
tail length rather than treating GR as a protocol-independent generalization
measure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inv_arc_test import _pick_device  # noqa: E402
from network_stats.run_one import (  # noqa: E402
    make_gr_input_stream_sweep,
    run_reservoir_stream_batch,
)
from network_stats.stats import compute_GR  # noqa: E402
from Parameter_justification.ipc_order_sweep_ce import (  # noqa: E402
    _build_model_reservoir,
    _expand_models,
    _make_neuron_bias,
    _model_title,
    _resolve_seeds,
)
from reservoir_variants import DEFAULT_SIM_PARAMS  # noqa: E402
from util.util import (  # noqa: E402
    assign_random_unknown_signs,
    load_connectome,
    load_unknown_sign_weights,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep the common-tail length used by the production GR path."
    )
    p.add_argument("--ce-adj", default="Connectome/ce_adj.npy")
    p.add_argument("--ce-ei", default="Connectome/ce_ei.npy")
    p.add_argument("--ce-unknown-sign-weights", default=None)
    p.add_argument(
        "--unknown-sign-policy",
        choices=("drop", "random_unknown_sign_matched"),
        default="drop",
    )
    p.add_argument("--unknown-sign-seed-offset", type=int, default=23_000_000)
    p.add_argument("--out-dir", default="gr_common_tail_sensitivity")

    p.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Reservoir models to compare. Use 'all' for the main architecture set.",
    )
    p.add_argument("--er-p", type=float, default=0.1)
    p.add_argument("--ws-p", type=float, default=0.1)

    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--rho", type=float, nargs="+", default=[0.95])
    p.add_argument("--leak", type=float, nargs="+", default=[0.8])
    p.add_argument("--input-scale", type=float, nargs="+", default=[1.0])
    p.add_argument("--neuron-biases", type=float, nargs="+", default=[0.0])

    p.add_argument(
        "--num-streams",
        type=int,
        default=DEFAULT_SIM_PARAMS.gr_num_streams,
    )
    p.add_argument(
        "--prefix-length",
        type=int,
        default=(
            DEFAULT_SIM_PARAMS.gr_stream_length
            - DEFAULT_SIM_PARAMS.gr_common_tail_length
        ),
        help="Fixed independently sampled prefix length for every tail condition.",
    )
    p.add_argument(
        "--common-tail-lengths",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 7, DEFAULT_SIM_PARAMS.gr_stream_length],
        help="Shared-continuation lengths to evaluate; 0 is a diagnostic control.",
    )
    p.add_argument(
        "--reference-tail-length",
        type=int,
        default=DEFAULT_SIM_PARAMS.gr_common_tail_length,
    )
    p.add_argument(
        "--gr-seed-offset",
        type=int,
        default=DEFAULT_SIM_PARAMS.gr_seed_offset,
    )
    p.add_argument("--cuda", type=int, default=None)
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> list[int]:
    if args.num_streams < 2:
        raise ValueError("--num-streams must be >= 2.")
    if args.prefix_length < 1:
        raise ValueError("--prefix-length must be >= 1.")
    if any(rho <= 0.0 for rho in args.rho):
        raise ValueError("--rho values must be positive.")
    if any(not 0.0 < leak <= 1.0 for leak in args.leak):
        raise ValueError("--leak values must lie in (0, 1].")
    if any(scale < 0.0 for scale in args.input_scale):
        raise ValueError("--input-scale values must be non-negative.")
    if any(bias < 0.0 for bias in args.neuron_biases):
        raise ValueError("--neuron-biases values must be non-negative.")

    tail_lengths = sorted(set(int(length) for length in args.common_tail_lengths))
    if any(length < 0 for length in tail_lengths):
        raise ValueError("--common-tail-lengths must be non-negative.")
    if args.reference_tail_length < 0:
        raise ValueError("--reference-tail-length must be non-negative.")
    if args.reference_tail_length not in tail_lengths:
        tail_lengths.append(args.reference_tail_length)
        tail_lengths.sort()
    return tail_lengths


def _correlation(current: pd.Series, reference: pd.Series, method: str) -> float:
    paired = pd.concat((current, reference), axis=1).dropna()
    if (
        len(paired) < 2
        or paired.iloc[:, 0].nunique() < 2
        or paired.iloc[:, 1].nunique() < 2
    ):
        return np.nan
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method=method))


def _summarize(
    raw: pd.DataFrame,
    tail_lengths: list[int],
    reference_tail_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        raw.groupby(["model", "common_tail_length"], as_index=False)
        .agg(
            mean_GR=("GR", "mean"),
            std_GR=("GR", lambda values: float(np.std(values.to_numpy()))),
            median_GR=("GR", "median"),
            n_runs=("GR", "size"),
        )
        .sort_values(["model", "common_tail_length"])
    )

    model_means = summary.pivot(
        index="model", columns="common_tail_length", values="mean_GR"
    )
    reference = model_means[reference_tail_length]
    rank_rows = []
    for tail_length in tail_lengths:
        current = model_means[tail_length]
        n_models = int(pd.concat((current, reference), axis=1).dropna().shape[0])
        if tail_length == reference_tail_length:
            pearson = spearman = 1.0
        else:
            pearson = _correlation(current, reference, "pearson")
            spearman = _correlation(current, reference, "spearman")
        rank_rows.append(
            {
                "common_tail_length": tail_length,
                "reference_tail_length": reference_tail_length,
                "pearson_model_means_vs_reference": pearson,
                "spearman_model_means_vs_reference": spearman,
                "n_models": n_models,
            }
        )
    return summary, pd.DataFrame(rank_rows)


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    models: list[str],
    seeds: list[int],
    tail_lengths: list[int],
    summary: pd.DataFrame,
    rank_stability: pd.DataFrame,
) -> None:
    lines = [
        "GR common-tail sensitivity report",
        "",
        (
            f"Protocol: {args.num_streams} streams with a fixed "
            f"{args.prefix_length}-step varying prefix; shared-tail "
            f"lengths={tail_lengths}; reference={args.reference_tail_length}."
        ),
        f"Models={models}; seeds={seeds}.",
        (
            "The prefix and shared-continuation onset are fixed across tail "
            "conditions. Total stream length is prefix_length + tail_length."
        ),
        (
            "Input construction reuses make_gr_input_streams; state evolution "
            "and GR reuse run_reservoir_stream_batch and compute_GR."
        ),
        "",
        (
            "Interpretation: lower GR means lower-dimensional final-state variation "
            "for streams sharing the specified recent input tail. It is an "
            "operational, protocol-dependent result, not a protocol-independent "
            "measure of task generalization."
        ),
        (
            "Tail length 0 is an independent-input diagnostic. Judge robustness "
            "from architecture ordering across positive tail lengths, not equality "
            "of absolute GR values."
        ),
        "",
        "[Architecture-order stability vs reference]",
    ]

    for row in rank_stability.itertuples(index=False):
        lines.append(
            f"tail={row.common_tail_length}: "
            f"pearson={row.pearson_model_means_vs_reference:.4f}, "
            f"spearman={row.spearman_model_means_vs_reference:.4f}, "
            f"n_models={row.n_models}"
        )

    lines.extend(["", "[Per-model GR means]"])
    for model in models:
        model_rows = summary[summary["model"] == model]
        values = ", ".join(
            f"tail={int(row.common_tail_length)}: {row.mean_GR:.4f}±{row.std_GR:.4f}"
            for row in model_rows.itertuples(index=False)
        )
        lines.append(f"{_model_title(model, args.er_p)}: {values}")

    lines.extend(
        [
            "",
            "This report quantifies sensitivity; it does not assume the result is tail-invariant.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tail_lengths = _validate_args(args)
    models = _expand_models(list(args.models))
    seeds = _resolve_seeds(args)
    device = _pick_device(args.cuda)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ce_W_bio, _, _ = load_connectome(args.ce_adj, args.ce_ei)
    if ce_W_bio is None:
        raise FileNotFoundError(f"Could not load CE adjacency from: {args.ce_adj}")

    unknown_sign_weights = None
    if args.unknown_sign_policy == "random_unknown_sign_matched":
        unknown_sign_weights = load_unknown_sign_weights(
            args.ce_adj,
            args.ce_unknown_sign_weights,
            n_nodes=ce_W_bio.shape[0],
        )
        if unknown_sign_weights is None:
            raise FileNotFoundError(
                "random_unknown_sign_matched requires an unknown-sign weight matrix."
            )

    rows: list[dict] = []
    for seed in seeds:
        set_seed(seed)
        ce_W_trial = ce_W_bio
        if args.unknown_sign_policy == "random_unknown_sign_matched":
            ce_W_trial = assign_random_unknown_signs(
                ce_W_bio,
                unknown_sign_weights,
                np.random.default_rng(seed + args.unknown_sign_seed_offset),
            )
        nnz_target_ce = int(np.count_nonzero(ce_W_trial))
        gr_seed = seed + args.gr_seed_offset
        stream_cache: dict[tuple[int, torch.dtype], dict[int, torch.Tensor]] = {}

        for rho in args.rho:
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
                    initial_state = torch.zeros(
                        Wt.shape[0], device=device, dtype=Wt.dtype
                    )

                    for leak in args.leak:
                        for neuron_bias in args.neuron_biases:
                            bias = _make_neuron_bias(
                                Wt.shape[0],
                                neuron_bias,
                                device,
                                Wt.dtype,
                                seed + 17_000_003,
                            )
                            cache_key = (int(Win.shape[1]), Win.dtype)
                            if cache_key not in stream_cache:
                                generator = torch.Generator(device=device)
                                generator.manual_seed(gr_seed)
                                stream_cache[cache_key] = make_gr_input_stream_sweep(
                                    n_streams=args.num_streams,
                                    prefix_length=args.prefix_length,
                                    common_tail_lengths=tail_lengths,
                                    reference_tail_length=args.reference_tail_length,
                                    n_inputs=int(Win.shape[1]),
                                    device=device,
                                    dtype=Win.dtype,
                                    generator=generator,
                                )
                            for tail_length in tail_lengths:
                                states = run_reservoir_stream_batch(
                                    Wt,
                                    Win,
                                    stream_cache[cache_key][tail_length],
                                    leak,
                                    initial_state=initial_state,
                                    bias=bias,
                                )
                                rows.append(
                                    {
                                        "model": model,
                                        "seed": seed,
                                        "rho_target": rho,
                                        "leak": leak,
                                        "input_scale": input_scale,
                                        "neuron_bias": neuron_bias,
                                        "num_streams": args.num_streams,
                                        "prefix_length": args.prefix_length,
                                        "common_tail_length": tail_length,
                                        "total_stream_length": (
                                            args.prefix_length + tail_length
                                        ),
                                        "gr_seed": gr_seed,
                                        "GR": compute_GR(states),
                                    }
                                )

    raw = pd.DataFrame(rows)
    summary, rank_stability = _summarize(raw, tail_lengths, args.reference_tail_length)

    raw_csv = out_dir / "gr_common_tail_sensitivity_raw.csv"
    summary_csv = out_dir / "gr_common_tail_sensitivity_summary.csv"
    rank_csv = out_dir / "gr_common_tail_rank_stability.csv"
    report_txt = out_dir / "gr_common_tail_sensitivity_report.txt"

    raw.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    rank_stability.to_csv(rank_csv, index=False)
    _write_report(
        report_txt,
        args=args,
        models=models,
        seeds=seeds,
        tail_lengths=tail_lengths,
        summary=summary,
        rank_stability=rank_stability,
    )

    for path in (raw_csv, summary_csv, rank_csv, report_txt):
        print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
