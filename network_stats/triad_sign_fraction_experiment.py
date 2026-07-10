#!/usr/bin/env python3
"""Triad weight summaries for sign-fraction post-normalized networks."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

from inv_arc_test import NORMALIZATION_MODES, SWEEP_SR, _split_indices
from util.util import (
    _count_edges,
    assign_random_unknown_signs,
    build_reservoir,
    load_connectome,
    load_connectome_node_names,
    load_unknown_sign_weights,
    negative_edge_fraction,
)


DEFAULT_SIGN_FRACS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_RHO_VALUES = tuple(SWEEP_SR)
TRIAD_SCOPE = "closed"


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    job: str
    unknown_sign_policy: str
    ce_adj: str
    ce_unknown_sign_weights: str | None = None


DEFAULT_DATASETS = {
    "new_cel_4to1": DatasetSpec(
        label="new_cel_4to1",
        job="sign_test_og_cel",
        unknown_sign_policy="random_unknown_4to1",
        ce_adj="Connectome/ce_adj_unk41.npy",
        ce_unknown_sign_weights="Connectome/ce_unknown_sign_weights_unk41.npy",
    ),
    "new_cel_removed": DatasetSpec(
        label="new_cel_removed",
        job="sign_test_og_cel",
        unknown_sign_policy="drop",
        ce_adj="Connectome/ce_adj_removed.npy",
    ),
    "matched_er_4to1": DatasetSpec(
        label="matched_er_4to1",
        job="sign_test_er",
        unknown_sign_policy="random_unknown_4to1",
        ce_adj="Connectome/ce_adj_unk41.npy",
        ce_unknown_sign_weights="Connectome/ce_unknown_sign_weights_unk41.npy",
    ),
}


def _sign_fractions_for_job(
    job: str,
    W_trial: np.ndarray,
    base_fracs,
    include_empirical_cel_fraction: bool,
) -> list[tuple[float, str]]:
    rows = [(float(frac), "requested") for frac in base_fracs]
    if job == "sign_test_og_cel" and include_empirical_cel_fraction:
        empirical = negative_edge_fraction(W_trial)
        if np.isfinite(empirical) and not any(np.isclose(empirical, frac) for frac, _ in rows):
            rows.insert(0, (empirical, "empirical"))
    return rows


def _candidate_triples(mask: np.ndarray) -> np.ndarray:
    """Unique unordered 3-node sets containing at least one directed edge."""
    n_nodes = mask.shape[0]
    triples: set[tuple[int, int, int]] = set()
    for i, j in np.argwhere(mask):
        for k in range(n_nodes):
            if k == i or k == j:
                continue
            triples.add(tuple(sorted((int(i), int(j), int(k)))))
    if not triples:
        return np.empty((0, 3), dtype=np.int32)
    return np.array(sorted(triples), dtype=np.int32)


def _summarize_values(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def triad_weight_summaries(
    W: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    """Summarize closed induced 3-node subgraphs after global weight normalization."""
    W_abs = np.abs(W.astype(np.float64, copy=False))
    mask = W_abs > 0
    np.fill_diagonal(mask, False)

    triples = _candidate_triples(mask)
    details: dict[str, np.ndarray] = {
        "triples": triples,
        "edge_count": np.empty(0, dtype=np.int16),
        "pair_count": np.empty(0, dtype=np.int16),
        "avg_abs_w": np.empty(0, dtype=np.float64),
        "cv_abs_w": np.empty(0, dtype=np.float64),
    }
    if triples.size == 0:
        row = {
            "triad_scope": TRIAD_SCOPE,
            "triad_count": 0,
            "triad_edge_count_mean": float("nan"),
            "triad_edge_count_std": float("nan"),
        }
        row.update(_summarize_values(np.empty(0), "triad_avg_abs_w"))
        row.update(_summarize_values(np.empty(0), "triad_cv_abs_w"))
        return [row], details

    a = triples[:, 0]
    b = triples[:, 1]
    c = triples[:, 2]
    vals = np.column_stack(
        (
            W_abs[a, b],
            W_abs[b, a],
            W_abs[a, c],
            W_abs[c, a],
            W_abs[b, c],
            W_abs[c, b],
        )
    )
    present = vals > 0
    edge_count = present.sum(axis=1).astype(np.int16)
    pair_count = np.column_stack(
        (
            present[:, 0] | present[:, 1],
            present[:, 2] | present[:, 3],
            present[:, 4] | present[:, 5],
        )
    ).sum(axis=1).astype(np.int16)

    edge_sum = vals.sum(axis=1)
    avg_abs_w = edge_sum / edge_count
    centered = np.where(present, vals - avg_abs_w[:, None], 0.0)
    within_std = np.sqrt(np.sum(centered * centered, axis=1) / edge_count)
    cv_abs_w = np.divide(
        within_std,
        avg_abs_w,
        out=np.zeros_like(within_std),
        where=avg_abs_w > 0,
    )

    details = {
        "triples": triples,
        "edge_count": edge_count,
        "pair_count": pair_count,
        "avg_abs_w": avg_abs_w,
        "cv_abs_w": cv_abs_w,
    }

    keep = pair_count == 3
    scoped_edges = edge_count[keep].astype(np.float64)
    scoped_avg = avg_abs_w[keep]
    scoped_cv = cv_abs_w[keep]
    row = {
        "triad_scope": TRIAD_SCOPE,
        "triad_count": int(keep.sum()),
        "triad_edge_count_mean": float(np.mean(scoped_edges)) if scoped_edges.size else float("nan"),
        "triad_edge_count_std": (
            float(np.std(scoped_edges, ddof=1)) if scoped_edges.size > 1 else 0.0
        ),
    }
    row.update(_summarize_values(scoped_avg, "triad_avg_abs_w"))
    row.update(_summarize_values(scoped_cv, "triad_cv_abs_w"))
    return [row], details


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_detail_rows(
    path: Path,
    base_row: dict[str, object],
    details: dict[str, np.ndarray],
    node_names: list[str],
) -> None:
    triples = details["triples"]
    if triples.size == 0:
        return
    edge_count = details["edge_count"]
    pair_count = details["pair_count"]
    keep = pair_count == 3

    rows = []
    for idx in np.flatnonzero(keep):
        i, j, k = (int(v) for v in triples[idx])
        rows.append(
            {
                **base_row,
                "triad_scope": TRIAD_SCOPE,
                "node_i": i,
                "node_j": j,
                "node_k": k,
                "node_i_name": node_names[i],
                "node_j_name": node_names[j],
                "node_k_name": node_names[k],
                "triad_edge_count": int(edge_count[idx]),
                "triad_pair_count": int(pair_count[idx]),
                "triad_avg_abs_w": float(details["avg_abs_w"][idx]),
                "triad_cv_abs_w": float(details["cv_abs_w"][idx]),
            }
        )
    _append_rows(path, rows)


def _summarize_repeats(summary_csv: Path, grouped_csv: Path) -> None:
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    group_cols = ["dataset", "job", "normalization", "rho_target", "sign_frac", "triad_scope"]
    value_cols = [
        "n_edges",
        "negative_edge_fraction",
        "raw_rho",
        "ref_rho",
        "post_rho",
        "scale_factor",
        "triad_count",
        "triad_edge_count_mean",
        "triad_avg_abs_w_mean",
        "triad_avg_abs_w_std",
        "triad_avg_abs_w_min",
        "triad_avg_abs_w_max",
        "triad_cv_abs_w_mean",
        "triad_cv_abs_w_std",
    ]
    agg = (
        df.groupby(group_cols, dropna=False)[value_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    agg.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]
    grouped_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(grouped_csv, index=False)


def _all_csv_name(csv_name: str) -> str:
    path = Path(csv_name)
    return f"{path.stem}.ALL{path.suffix}"


def _merge_chunk_csvs(root: Path, csv_name: str, out_name: str) -> tuple[Path, int, int]:
    files = sorted(root.glob(f"chunk_*/{csv_name}"))
    if not files:
        raise FileNotFoundError(f"No chunk CSVs found matching {root}/chunk_*/{csv_name}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df.insert(0, "chunk", path.parent.name)
        frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame()

    out_path = root / out_name
    root.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return out_path, len(files), len(merged)


def merge_chunk_outputs(
    out_dir: Path,
    *,
    summary_csv_name: str,
    grouped_csv_name: str,
    detail_csv_name: str,
    merge_details: bool = False,
) -> None:
    merged_summary_name = _all_csv_name(summary_csv_name)
    merged_grouped_name = _all_csv_name(grouped_csv_name)
    merged_summary, n_summary_files, n_summary_rows = _merge_chunk_csvs(
        out_dir,
        summary_csv_name,
        merged_summary_name,
    )
    merged_grouped = out_dir / merged_grouped_name
    _summarize_repeats(merged_summary, merged_grouped)

    print(f"[info] merged {n_summary_files} summary chunk files")
    print(f"[info] wrote {merged_summary} ({n_summary_rows} rows)")
    print(f"[info] wrote {merged_grouped}")

    if merge_details:
        merged_detail_name = _all_csv_name(detail_csv_name)
        merged_detail, n_detail_files, n_detail_rows = _merge_chunk_csvs(
            out_dir,
            detail_csv_name,
            merged_detail_name,
        )
        print(f"[info] merged {n_detail_files} detail chunk files")
        print(f"[info] wrote {merged_detail} ({n_detail_rows} rows)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count triads and summarize |w| inside triads for the post-normalization "
            "sign-fraction networks used by the previous CE/ER sign experiments."
        )
    )
    p.add_argument("--out-dir", default="network_stats/triad_sign_fraction_results")
    p.add_argument("--summary-csv", default="triad_sign_fraction_summary.csv")
    p.add_argument("--grouped-csv", default="triad_sign_fraction_group_summary.csv")
    p.add_argument("--detail-csv", default="triad_sign_fraction_details.csv")
    p.add_argument(
        "--merge-chunks",
        action="store_true",
        help=(
            "Merge chunk_*/summary CSVs under --out-dir into *.ALL.csv files and "
            "write the final grouped summary, then exit."
        ),
    )
    p.add_argument(
        "--merge-details",
        action="store_true",
        help="With --merge-chunks, also merge chunk_*/detail CSVs. Detail files can be very large.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DEFAULT_DATASETS) + ("all",),
        default=("all",),
    )
    p.add_argument("--sign-fracs", type=float, nargs="+", default=DEFAULT_SIGN_FRACS)
    p.add_argument("--rho-values", type=float, nargs="+", default=DEFAULT_RHO_VALUES)
    p.add_argument(
        "--normalization-modes",
        nargs="+",
        choices=NORMALIZATION_MODES,
        default=("spectral_radius",),
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n-repeats", type=int, default=100)
    p.add_argument("--repeat-split", type=int, default=1)
    p.add_argument("--repeat-rank", type=int, default=0)
    p.add_argument("--repeat-seed-stride", type=int, default=100000)
    p.add_argument("--unknown-sign-inhibitory-frac", type=float, default=0.2)
    p.add_argument("--unknown-sign-seed-offset", type=int, default=23_000_000)
    p.add_argument(
        "--include-empirical-cel-fraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the actual CE negative-edge fraction inserted by sign_test_og_cel.",
    )
    p.add_argument("--er-p", type=float, default=0.1)
    p.add_argument(
        "--write-triad-details",
        action="store_true",
        help="Write one row per closed triad, including node indices/names. This can be large.",
    )
    p.add_argument(
        "--append-existing",
        action="store_true",
        help="Append to existing CSVs instead of replacing them before this run.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.merge_chunks:
        merge_chunk_outputs(
            out_dir,
            summary_csv_name=args.summary_csv,
            grouped_csv_name=args.grouped_csv,
            detail_csv_name=args.detail_csv,
            merge_details=args.merge_details,
        )
        return

    if any(frac < 0.0 or frac > 1.0 for frac in args.sign_fracs):
        raise ValueError("--sign-fracs must be in [0, 1].")
    if any(rho <= 0.0 for rho in args.rho_values):
        raise ValueError("--rho-values must be positive.")
    if not (0.0 <= args.unknown_sign_inhibitory_frac <= 1.0):
        raise ValueError("--unknown-sign-inhibitory-frac must be in [0, 1].")

    summary_csv = out_dir / args.summary_csv
    grouped_csv = out_dir / args.grouped_csv
    detail_csv = out_dir / args.detail_csv
    if not args.append_existing:
        for path in (summary_csv, grouped_csv, detail_csv):
            if path.exists():
                path.unlink()

    dataset_keys = list(DEFAULT_DATASETS) if "all" in args.datasets else list(args.datasets)
    repeat_ids = _split_indices(args.n_repeats, args.repeat_split, args.repeat_rank)
    device = torch.device("cpu")

    for dataset_key in dataset_keys:
        spec = DEFAULT_DATASETS[dataset_key]
        ce_W_base, _, _ = load_connectome(spec.ce_adj, None)
        if ce_W_base is None:
            raise FileNotFoundError(f"Could not load CE adjacency: {spec.ce_adj}")
        node_names = load_connectome_node_names(spec.ce_adj, ce_W_base.shape[0])
        unknown_weights = None
        if spec.unknown_sign_policy == "random_unknown_4to1":
            unknown_weights = load_unknown_sign_weights(
                spec.ce_adj,
                spec.ce_unknown_sign_weights,
                n_nodes=ce_W_base.shape[0],
            )
            if unknown_weights is None:
                raise FileNotFoundError(
                    f"{spec.label} requires unknown-sign weights for random_unknown_4to1."
                )

        for rep_idx in repeat_ids:
            seed_base = args.seed + rep_idx * args.repeat_seed_stride
            ce_W_trial = ce_W_base
            if spec.unknown_sign_policy == "random_unknown_4to1":
                ce_W_trial = assign_random_unknown_signs(
                    ce_W_base,
                    unknown_weights,
                    np.random.default_rng(seed_base + args.unknown_sign_seed_offset),
                    inhibitory_fraction=args.unknown_sign_inhibitory_frac,
                )

            sign_rows = _sign_fractions_for_job(
                spec.job,
                ce_W_trial,
                args.sign_fracs,
                args.include_empirical_cel_fraction,
            )
            for normalization, rho_target, (sign_frac, sign_frac_source) in itertools.product(
                args.normalization_modes,
                args.rho_values,
                sign_rows,
            ):
                if spec.job == "sign_test_er":
                    feature_conn = "sign_test_er"
                    nnz_target = _count_edges(ce_W_trial)
                    normalization_ref = ce_W_trial
                elif spec.job == "sign_test_og_cel":
                    feature_conn = "sign_test_og_cel"
                    nnz_target = None
                    normalization_ref = ce_W_trial
                else:
                    raise ValueError(f"Unsupported sign-fraction job: {spec.job}")

                Wt, _, norm_info = build_reservoir(
                    feature_conn=feature_conn,
                    target_sr=float(rho_target),
                    N=ce_W_trial.shape[0],
                    ce_W_bio=ce_W_trial,
                    input_scale=1.0,
                    seed=seed_base + 1,
                    nnz_target=nnz_target,
                    DEVICE=device,
                    per_neg=float(sign_frac),
                    normalization_mode=normalization,
                    normalization_ref=normalization_ref,
                    return_info=True,
                )
                W_post = Wt.detach().cpu().numpy()
                n_edges = int(np.count_nonzero(W_post))
                n_negative = int(np.count_nonzero(W_post < 0))
                base_row: dict[str, object] = {
                    "dataset": spec.label,
                    "job": spec.job,
                    "unknown_sign_policy": spec.unknown_sign_policy,
                    "normalization": normalization,
                    "rho_target": float(rho_target),
                    "repeat_id": int(rep_idx),
                    "seed": int(seed_base),
                    "sign_frac": float(sign_frac),
                    "sign_frac_source": sign_frac_source,
                    "n_nodes": int(W_post.shape[0]),
                    "n_edges": n_edges,
                    "negative_edges": n_negative,
                    "negative_edge_fraction": float(n_negative / n_edges) if n_edges else float("nan"),
                    "raw_rho": float(norm_info["raw_rho"]),
                    "ref_rho": float(norm_info["ref_rho"]),
                    "post_rho": float(norm_info["post_rho"]),
                    "scale_factor": float(norm_info["scale_factor"]),
                }

                triad_rows, details = triad_weight_summaries(W_post)
                rows = []
                for triad_row in triad_rows:
                    rows.append(
                        {
                            **base_row,
                            **triad_row,
                        }
                    )
                _append_rows(summary_csv, rows)
                if args.write_triad_details:
                    _write_detail_rows(
                        detail_csv,
                        base_row,
                        details,
                        node_names,
                    )

    _summarize_repeats(summary_csv, grouped_csv)
    print(f"[info] wrote {summary_csv}")
    print(f"[info] wrote {grouped_csv}")
    if args.write_triad_details:
        print(f"[info] wrote {detail_csv}")


if __name__ == "__main__":
    main()
