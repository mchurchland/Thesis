#!/usr/bin/env python3
"""Topology-only audit of the directed degree-matched shuffle.

This reconstructs the same sign-matched C. elegans matrix used by the
``new_cel_sign_matched`` experiments and runs only the topology shuffle.  It
does not instantiate a reservoir or compute any reservoir metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from util.util import (
    UNKNOWN_SIGN_INHIBITORY_FRACTION,
    assign_random_unknown_signs,
    degree_matched_shuffle_directed,
    load_connectome,
    load_unknown_sign_weights,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the exact directed degree-matched shuffle on the "
            "sign-matched C. elegans topology; no reservoir simulation is run."
        )
    )
    parser.add_argument(
        "--ce-adj",
        type=Path,
        default=ROOT / "Connectome" / "ce_adj.npy",
        help="Known-sign C. elegans adjacency matrix.",
    )
    parser.add_argument(
        "--ce-unknown-sign-weights",
        type=Path,
        default=ROOT / "Connectome" / "ce_unknown_sign_weights.npy",
        help="Magnitudes of the complex/no-prediction edges.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1000,
        help="Number of independent shuffle seeds (default: 1000).",
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=20_000,
        help="Maximum failed swap attempts per run, matching production.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=12_345,
        help="First production-style repeat seed.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=100_000,
        help="Stride between production-style repeat seeds.",
    )
    parser.add_argument(
        "--unknown-sign-seed-offset",
        type=int,
        default=23_000_000,
        help="Offset used when assigning formerly unknown edge signs.",
    )
    parser.add_argument(
        "--unknown-sign-inhibitory-frac",
        type=float,
        default=UNKNOWN_SIGN_INHIBITORY_FRACTION,
        help="Negative fraction for formerly unknown-sign edges.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "final_results" / "topology_shuffle_audit" / "cel_matched",
        help="Directory for per-run CSV and JSON summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N runs; use 0 to disable.",
    )
    args = parser.parse_args()

    if args.n_runs <= 0:
        parser.error("--n-runs must be positive")
    if args.tries < 0:
        parser.error("--tries must be non-negative")
    if args.seed_stride <= 0:
        parser.error("--seed-stride must be positive")
    if not 0.0 <= args.unknown_sign_inhibitory_frac <= 1.0:
        parser.error("--unknown-sign-inhibitory-frac must be in [0, 1]")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    return args


def describe(values: np.ndarray) -> dict[str, float]:
    """Return compact distribution summaries using JSON-native floats."""
    return {
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def load_cel_matched_inputs(
    ce_adj_path: Path,
    unknown_weights_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the two processed matrices used to construct ``cel_matched``."""
    known, _, _ = load_connectome(str(ce_adj_path), None)
    if known is None:
        raise FileNotFoundError(f"Could not load C. elegans adjacency: {ce_adj_path}")

    unknown = load_unknown_sign_weights(
        str(ce_adj_path),
        str(unknown_weights_path),
        n_nodes=known.shape[0],
    )
    if unknown is None:
        raise FileNotFoundError(
            f"Could not load unknown-sign edge weights: {unknown_weights_path}"
        )

    known_support = known != 0
    unknown_support = unknown != 0
    overlap = int(np.count_nonzero(known_support & unknown_support))
    if overlap:
        raise ValueError(
            "Known-sign and unknown-sign matrices overlap at "
            f"{overlap} edge positions; their topology is not a disjoint union."
        )

    return known, unknown


def main() -> None:
    args = parse_args()
    known, unknown = load_cel_matched_inputs(
        args.ce_adj,
        args.ce_unknown_sign_weights,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "degree_matched_shuffle_runs.csv"
    summary_path = args.out_dir / "degree_matched_shuffle_summary.json"

    known_edges = int(np.count_nonzero(known))
    unknown_edges = int(np.count_nonzero(unknown))
    expected_edges = known_edges + unknown_edges
    rows: list[dict[str, int | float | bool | str]] = []

    for run_index in range(args.n_runs):
        seed_base = args.base_seed + run_index * args.seed_stride
        unknown_sign_seed = seed_base + args.unknown_sign_seed_offset

        cel_matched = assign_random_unknown_signs(
            known,
            unknown,
            np.random.default_rng(unknown_sign_seed),
            inhibitory_fraction=args.unknown_sign_inhibitory_frac,
        )
        if int(np.count_nonzero(cel_matched)) != expected_edges:
            raise RuntimeError(
                "The reconstructed cel_matched matrix does not have the expected "
                f"{expected_edges} edges."
            )

        original_support = cel_matched != 0
        original_in_degree = np.count_nonzero(original_support, axis=0)
        original_out_degree = np.count_nonzero(original_support, axis=1)
        original_weights = np.sort(cel_matched[original_support])

        shuffled, stats = degree_matched_shuffle_directed(
            cel_matched,
            tries=args.tries,
            rng=np.random.default_rng(seed_base),
            return_stats=True,
        )

        shuffled_support = shuffled != 0
        in_degree_preserved = np.array_equal(
            original_in_degree,
            np.count_nonzero(shuffled_support, axis=0),
        )
        out_degree_preserved = np.array_equal(
            original_out_degree,
            np.count_nonzero(shuffled_support, axis=1),
        )
        edge_count_preserved = int(np.count_nonzero(shuffled_support)) == expected_edges
        diagonal_clear = not np.any(np.diag(shuffled_support))
        weight_multiset_preserved = np.array_equal(
            original_weights,
            np.sort(shuffled[shuffled_support]),
        )

        if not (
            in_degree_preserved
            and out_degree_preserved
            and edge_count_preserved
            and diagonal_clear
            and weight_multiset_preserved
        ):
            raise RuntimeError(
                f"Shuffle invariant failed for run {run_index} (seed {seed_base})."
            )

        rows.append(
            {
                "run_index": run_index,
                "shuffle_seed": seed_base,
                "unknown_sign_seed": unknown_sign_seed,
                **stats,
                "in_degree_preserved": in_degree_preserved,
                "out_degree_preserved": out_degree_preserved,
                "edge_count_preserved": edge_count_preserved,
                "diagonal_clear": diagonal_clear,
                "weight_multiset_preserved": weight_multiset_preserved,
            }
        )

        if args.progress_every and (
            (run_index + 1) % args.progress_every == 0
            or run_index + 1 == args.n_runs
        ):
            print(f"Completed {run_index + 1}/{args.n_runs} shuffle runs")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful_swaps = np.asarray(
        [row["successful_swaps"] for row in rows], dtype=np.float64
    )
    successful_fraction = np.asarray(
        [row["successful_fraction"] for row in rows], dtype=np.float64
    )
    failed_attempts = np.asarray(
        [row["failed_swap_attempts"] for row in rows], dtype=np.float64
    )
    retained_fraction = np.asarray(
        [row["retained_edge_fraction"] for row in rows], dtype=np.float64
    )
    reached_max = np.asarray(
        [row["reached_max_swaps"] for row in rows], dtype=bool
    )
    hit_failed_limit = np.asarray(
        [row["hit_failed_attempt_limit"] for row in rows], dtype=bool
    )

    summary = {
        "dataset": "cel_matched",
        "nodes": int(known.shape[0]),
        "known_sign_edges": known_edges,
        "formerly_unknown_sign_edges": unknown_edges,
        "edges": expected_edges,
        "one_pass_max_swaps": expected_edges // 2,
        "n_runs": args.n_runs,
        "tries": args.tries,
        "base_seed": args.base_seed,
        "seed_stride": args.seed_stride,
        "unknown_sign_seed_offset": args.unknown_sign_seed_offset,
        "unknown_sign_inhibitory_fraction": args.unknown_sign_inhibitory_frac,
        "runs_reaching_max_swaps": int(np.count_nonzero(reached_max)),
        "fraction_reaching_max_swaps": float(np.mean(reached_max)),
        "runs_hitting_failed_attempt_limit": int(
            np.count_nonzero(hit_failed_limit)
        ),
        "fraction_hitting_failed_attempt_limit": float(
            np.mean(hit_failed_limit)
        ),
        "all_graph_invariants_passed": True,
        "successful_swaps": describe(successful_swaps),
        "successful_fraction": describe(successful_fraction),
        "failed_swap_attempts": describe(failed_attempts),
        "retained_edge_fraction": describe(retained_fraction),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print()
    print(f"Dataset: cel_matched ({known.shape[0]} nodes, {expected_edges} edges)")
    print(f"One-pass maximum: {expected_edges // 2} successful swaps")
    print(
        "Reached maximum: "
        f"{summary['runs_reaching_max_swaps']}/{args.n_runs} "
        f"({summary['fraction_reaching_max_swaps']:.1%})"
    )
    print(
        "Hit failed-attempt limit: "
        f"{summary['runs_hitting_failed_attempt_limit']}/{args.n_runs} "
        f"({summary['fraction_hitting_failed_attempt_limit']:.1%})"
    )
    print(
        "Successful swaps (min/median/max): "
        f"{successful_swaps.min():.0f}/"
        f"{np.median(successful_swaps):.0f}/"
        f"{successful_swaps.max():.0f}"
    )
    print(
        "Failed attempts (min/median/max): "
        f"{failed_attempts.min():.0f}/"
        f"{np.median(failed_attempts):.0f}/"
        f"{failed_attempts.max():.0f}"
    )
    print(
        "Original edges retained (mean): "
        f"{retained_fraction.mean():.3%}"
    )
    print(f"Per-run results: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
