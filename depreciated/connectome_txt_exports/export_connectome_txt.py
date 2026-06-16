#!/usr/bin/env python3
"""Export a C. elegans adjacency matrix to text, optionally with sign flips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _normalize_fraction(value: str) -> float:
    """Accept fractions in [0, 1] or percentages in (1, 100]."""
    try:
        value_float = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sign flip fraction must be numeric.") from exc

    value = value_float
    if 0.0 <= value <= 1.0:
        return value
    if 1.0 < value <= 100.0:
        return value / 100.0
    raise argparse.ArgumentTypeError("sign flip fraction must be in [0, 1] or percentage in (1, 100].")


def _sign_counts(W: np.ndarray) -> tuple[int, int, int]:
    pos = int((W > 0).sum())
    neg = int((W < 0).sum())
    return pos, neg, pos + neg


def _write_matrix_txt(path: Path, W: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, W.astype(np.float32, copy=False), fmt="%.8g", header=header)


def _flip_fraction_of_nonzero_signs(
    W: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    out = W.copy().astype(np.float32, copy=False)
    nz = np.argwhere(out != 0)
    n_flip = int(round(frac * len(nz)))
    if n_flip == 0:
        return out, 0

    selected = rng.choice(len(nz), size=n_flip, replace=False)
    rows = nz[selected, 0]
    cols = nz[selected, 1]
    out[rows, cols] *= -1.0
    return out, n_flip


def _print_summary(label: str, path: Path, W: np.ndarray) -> None:
    pos, neg, nz = _sign_counts(W)
    p_neg = neg / nz if nz else float("nan")
    print(
        f"{label}: wrote {path} | shape={W.shape} pos={pos} neg={neg} "
        f"nonzero={nz} p_neg={p_neg:.6f}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export a C. elegans adjacency .npy matrix to .txt, optionally flipping signs."
    )
    ap.add_argument(
        "--ce-adj",
        default="Connectome/ce_adj_new.npy",
        help="Input C. elegans adjacency .npy path.",
    )
    ap.add_argument(
        "--out-dir",
        default="connectome_txt_exports",
        help="Directory for exported .txt matrices.",
    )
    ap.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input adjacency stem.",
    )
    ap.add_argument(
        "--original-connectome",
        action="store_true",
        help="Export the loaded connectome unchanged.",
    )
    ap.add_argument(
        "--sign-flip-frac",
        type=_normalize_fraction,
        default=None,
        help="Fraction of nonzero edge signs to flip. Accepts 0.25 or 25 for 25 percent.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="RNG seed used when --sign-flip-frac is passed.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.original_connectome and args.sign_flip_frac is None:
        raise SystemExit("Pass --original-connectome, --sign-flip-frac, or both.")

    adj_path = Path(args.ce_adj)
    W = np.load(adj_path).astype(np.float32, copy=False)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("Input adjacency must be a square 2D matrix.")

    out_dir = Path(args.out_dir)
    prefix = args.prefix or adj_path.stem

    if args.original_connectome:
        out_path = out_dir / f"{prefix}_original.txt"
        pos, neg, nz = _sign_counts(W)
        header = f"source={adj_path} original shape={W.shape} pos={pos} neg={neg} nonzero={nz}"
        _write_matrix_txt(out_path, W, header)
        _print_summary("original", out_path, W)

    if args.sign_flip_frac is not None:
        rng = np.random.default_rng(args.seed)
        W_flip, n_flip = _flip_fraction_of_nonzero_signs(W, args.sign_flip_frac, rng)
        frac_tag = f"{args.sign_flip_frac:g}".replace(".", "p")
        out_path = out_dir / f"{prefix}_signflip_{frac_tag}.txt"
        pos, neg, nz = _sign_counts(W_flip)
        header = (
            f"source={adj_path} sign_flip_frac={args.sign_flip_frac:g} seed={args.seed} "
            f"flipped_edges={n_flip} shape={W_flip.shape} pos={pos} neg={neg} nonzero={nz}"
        )
        _write_matrix_txt(out_path, W_flip, header)
        _print_summary(f"sign-flipped ({n_flip} edges)", out_path, W_flip)


if __name__ == "__main__":
    main()
