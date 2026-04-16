#!/usr/bin/env python3
"""Find C. elegans neurons with zero in-degree in the connectome."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_connectome_with_names(adj_path: str) -> tuple[np.ndarray, list[str]]:
    adj_file = Path(adj_path)
    W_bio = np.load(adj_file).astype(np.float32, copy=False)
    if W_bio.ndim != 2 or W_bio.shape[0] != W_bio.shape[1]:
        raise ValueError("CE adjacency must be a square 2D array.")

    if not np.isfinite(W_bio).all():
        W_bio = np.where(np.isfinite(W_bio), W_bio, 0.0).astype(np.float32, copy=False)
    np.fill_diagonal(W_bio, 0.0)

    names_path = adj_file.with_name("ce_nodes.txt")
    if names_path.is_file():
        with names_path.open("r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
        if len(names) != W_bio.shape[0]:
            raise ValueError("Names length must equal adjacency size.")
    else:
        names = [str(i) for i in range(W_bio.shape[0])]

    return W_bio, names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print neurons with zero in-degree in the C. elegans connectome."
    )
    parser.add_argument("--adj", default="Connectome/ce_adj.npy", help="Adjacency .npy path.")
    args = parser.parse_args()

    W_bio, names = load_connectome_with_names(args.adj)
    indegree = np.count_nonzero(W_bio, axis=0)
    zero_indegree_idx = np.flatnonzero(indegree == 0)

    print(f"Loaded adjacency: {args.adj}")
    print(f"Nodes: {W_bio.shape[0]}")
    print(f"Neurons with zero in-degree: {len(zero_indegree_idx)}")

    for idx in zero_indegree_idx:
        print(f"{idx}\t{names[idx]}")


if __name__ == "__main__":
    main()
