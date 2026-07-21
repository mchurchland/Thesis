#!/usr/bin/env python3
"""Summarize the spectral radii actually used by the sign sweeps.

For each dataset, normalization mode, negative-edge fraction, and nominal
spectral-radius target, the script first averages duplicate hyperparameter rows
within each repeat and then summarizes the achieved ``post_rho`` across repeats.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUTS = (
    Path("final_results/sign_norm/cel_matched/combined.ALL.csv"),
    Path("final_results/sign_norm/cel_removed/combined.ALL.csv"),
    Path("final_results/sign_norm/matched_er/combined.ALL.csv"),
)
DEFAULT_OUTPUT = Path("final_results/sign_norm/effective_rho_sweep_values.csv")

_NUMBER_BEFORE_NORM = re.compile(
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?=__norm_)"
)
_NORM_SUFFIX = re.compile(r"__norm_([A-Za-z0-9_+-]+)$")


def _parse_sign_fraction(mode: pd.Series) -> pd.Series:
    return pd.to_numeric(
        mode.astype(str).str.extract(_NUMBER_BEFORE_NORM, expand=False),
        errors="coerce",
    )


def _parse_normalization(mode: pd.Series) -> pd.Series:
    return mode.astype(str).str.extract(_NORM_SUFFIX, expand=False).fillna("")


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    required = {
        "mode",
        "rho_target",
        "post_rho",
        "raw_rho",
        "ref_rho",
        "src",
        "shuffle_id",
        "seed",
    }
    header = pd.read_csv(path, nrows=0)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    usecols = sorted(required.union({"normalization"}).intersection(header.columns))
    frame = pd.read_csv(path, usecols=usecols)
    frame["dataset"] = path.parent.name
    frame["sign_fraction"] = _parse_sign_fraction(frame["mode"])

    inferred_norm = _parse_normalization(frame["mode"])
    if "normalization" not in frame.columns:
        frame["normalization"] = inferred_norm
    else:
        frame["normalization"] = frame["normalization"].fillna("").astype(str).str.strip()
        missing_norm = frame["normalization"].isin(("", "nan", "None"))
        frame.loc[missing_norm, "normalization"] = inferred_norm.loc[missing_norm]

    for column in ("rho_target", "post_rho", "raw_rho", "ref_rho"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["sign_fraction", "rho_target", "post_rho"])
    frame["repeat_id"] = (
        frame["src"].astype(str)
        + ":"
        + frame["shuffle_id"].astype(str)
        + ":"
        + frame["seed"].astype(str)
    )
    return frame


def summarize(inputs: list[Path], normalization: str) -> pd.DataFrame:
    data = pd.concat((_read_input(path) for path in inputs), ignore_index=True)
    if normalization != "all":
        data = data[data["normalization"] == normalization].copy()
    if data.empty:
        raise ValueError(f"No rows found for normalization={normalization!r}.")

    group_keys = [
        "dataset",
        "normalization",
        "sign_fraction",
        "rho_target",
        "repeat_id",
    ]
    # post_rho is repeated for every leak/input/bias combination. Collapse those
    # duplicates so incomplete hyperparameter grids cannot overweight a repeat.
    per_repeat = (
        data.groupby(group_keys, as_index=False)[["post_rho", "raw_rho", "ref_rho"]]
        .mean()
        .sort_values(group_keys)
    )

    summary_keys = ["dataset", "normalization", "sign_fraction", "rho_target"]
    result = (
        per_repeat.groupby(summary_keys, as_index=False)
        .agg(
            mean_effective_rho=("post_rho", "mean"),
            sd_effective_rho=("post_rho", "std"),
            n_repeats=("post_rho", "count"),
            mean_raw_rho=("raw_rho", "mean"),
            mean_reference_rho=("ref_rho", "mean"),
        )
        .sort_values(summary_keys)
        .reset_index(drop=True)
    )
    result["sd_effective_rho"] = result["sd_effective_rho"].fillna(0.0)
    result["sem_effective_rho"] = result["sd_effective_rho"] / np.sqrt(
        result["n_repeats"].clip(lower=1)
    )
    return result


def print_sweep_tables(summary: pd.DataFrame) -> None:
    for (dataset, normalization), subset in summary.groupby(
        ["dataset", "normalization"], sort=True
    ):
        wide = subset.pivot(
            index="sign_fraction",
            columns="rho_target",
            values="mean_effective_rho",
        ).sort_index()
        wide.columns = [f"nominal_{target:g}" for target in wide.columns]
        wide["effective_min"] = wide.min(axis=1)
        wide["effective_max"] = wide.max(axis=1)
        print(f"\n{dataset} — {normalization}")
        print(wide.to_string(float_format=lambda value: f"{value:.6f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_INPUTS),
        help=(
            "Merged sign-normalization CSV files. By default, use cel_matched, "
            "cel_removed, and matched_er under final_results/sign_norm."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=("original_radius", "spectral_radius", "all"),
        default="original_radius",
        help="Normalization mode to summarize (default: original_radius).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Long-form output CSV (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.inputs, args.normalization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print_sweep_tables(summary)
    print(f"\nSaved long-form summary to {args.output}")


if __name__ == "__main__":
    main()
