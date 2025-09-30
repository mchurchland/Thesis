# build_ce_connectome.py
# ------------------------------------------------------------
# Parse CElegansNeuronTables.xls and produce:
#   ce_adj.npy        (float32 NxN adjacency with signed synapse counts)
#   ce_adj_sr099.npy  (same, scaled to spectral radius ~0.99)
#   ce_nodes.txt      (node names, one per line, same order as matrix)
#   ce_ei.npy         (length N, +1=exc, -1=inh, 0=unknown inferred from outgoing edges)
#
# Sheets expected:
#   - "Connectome"         with columns: Origin, Target, Number of Connections, Neurotransmitter
#   - "NeuronsToMuscle"    with columns: Neuron, Muscle, Number of Connections, Neurotransmitter
#
# Notes
# - Edge sign rule: GABA → negative; ACh/Glutamate → positive; others default to + (changeable).
# - Duplicate edges are summed.
# - Muscles can be included or dropped via flag.
# - Dale label per neuron is inferred from the sign of its outgoing weights; near-zero → 0.
#
# Usage:
#   python build_ce_connectome.py --xls CElegansNeuronTables.xls --out ce \
#       --include-muscles  # drop this flag to exclude muscles
#
# Requirements:
#   pip/conda: pandas numpy xlrd (for .xls)  OR openpyxl (for .xlsx with engine="openpyxl")
# ------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Tuple, List, Optional

# ------------- helpers -------------

def canonical_nt(x: str) -> str:
    """Normalize neurotransmitter strings to coarse labels."""
    if not isinstance(x, str):
        return "UNK"
    s = x.strip().lower()
    # common variants
    if s in {"gaba"}:
        return "GABA"
    if s in {"acetylcholine", "ach", "aCh".lower()}:
        return "ACh"
    if s in {"glutamate", "glu", "glutamatergic"}:
        return "GLU"
    if s in {"serotonin", "5-ht", "5ht"}:
        return "5HT"
    if s in {"dopamine", "da"}:
        return "DA"
    if s in {"octopamine"}:
        return "OCT"
    if s in {"tyramine"}:
        return "TYR"
    if s in {"peptide", "neuropeptide", "np"}:
        return "PEP"
    if s in {"unknown", "unk"}:
        return "UNK"
    # fallback: keep original upper
    return x.strip().upper()

def nt_to_edge_sign(nt: str, default_pos_if_unknown: bool = True) -> int:
    """
    Map neurotransmitter → edge sign.
    GABA -> -1; ACh/GLU -> +1; others → +1 by default (set default_pos_if_unknown=False to make 0).
    """
    c = canonical_nt(nt)
    if c == "GABA":
        return -1
    if c in {"ACh", "GLU"}:
        return +1
    return +1 if default_pos_if_unknown else 0

def add_edge(acc: Dict[str, Dict[str, float]], src: str, dst: str, w: float):
    if src not in acc:
        acc[src] = {}
    acc[src][dst] = acc[src].get(dst, 0.0) + float(w)

def spectral_radius_numpy(W: np.ndarray) -> float:
    if W.size == 0:
        return 0.0
    # power iteration for stability
    v = np.random.default_rng(0).standard_normal(W.shape[0]).astype(np.float64)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(100):
        v = W @ v
        n = np.linalg.norm(v)
        if n < 1e-12:
            break
        v /= n
    lam = float(v @ (W @ v) / (v @ v + 1e-12))
    return abs(lam)

def scale_to_sr(W: np.ndarray, target: float = 0.99) -> np.ndarray:
    sr = spectral_radius_numpy(W.astype(np.float64))
    if sr < 1e-12:
        return W.copy()
    return (target / sr) * W

# ------------- core parsing -------------

def read_sheet(path: str, sheet: str, engine: Optional[str]) -> pd.DataFrame:
    # If engine not provided, try xlrd for .xls, else openpyxl
    if engine is None:
        engine = "xlrd" if path.lower().endswith(".xls") else "openpyxl"
    return pd.read_excel(path, sheet_name=sheet, engine=engine)

def process_connectome(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Expect columns: Origin, Target, Number of Connections, Neurotransmitter
    Returns nested dict: from -> {to: signed_weight}
    """
    # normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    origin = cols.get("origin")
    target = cols.get("target")
    num    = cols.get("number of connections") or cols.get("number_of_connections")
    nt     = cols.get("neurotransmitter")

    if not all([origin, target, num, nt]):
        raise ValueError("Connectome sheet must have columns: Origin, Target, Number of Connections, Neurotransmitter")

    acc: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        src = str(row[origin]).strip()
        dst = str(row[target]).strip()
        if src == "" or dst == "" or pd.isna(row[num]):
            continue
        weight = float(row[num])
        sign = nt_to_edge_sign(row[nt])
        add_edge(acc, src, dst, sign * weight)
    return acc

def process_neuron_to_muscle(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Expect columns: Neuron, Muscle, Number of Connections, Neurotransmitter
    """
    cols = {c.lower().strip(): c for c in df.columns}
    neuron = cols.get("neuron")
    muscle = cols.get("muscle")
    num    = cols.get("number of connections") or cols.get("number_of_connections")
    nt     = cols.get("neurotransmitter")

    if not all([neuron, muscle, num, nt]):
        raise ValueError("NeuronsToMuscle sheet must have Neuron, Muscle, Number of Connections, Neurotransmitter")

    acc: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        src = str(row[neuron]).strip()
        dst = str(row[muscle]).strip()
        if src == "" or dst == "" or pd.isna(row[num]):
            continue
        weight = float(row[num])
        sign = nt_to_edge_sign(row[nt])
        add_edge(acc, src, dst, sign * weight)
    return acc

def merge_edge_maps(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {k: v.copy() for k, v in a.items()}
    for src, d in b.items():
        if src not in out:
            out[src] = {}
        for dst, w in d.items():
            out[src][dst] = out[src].get(dst, 0.0) + float(w)
    return out

def build_matrix(edge_map: Dict[str, Dict[str, float]], include_nodes: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    nodes = set()
    for s, dd in edge_map.items():
        nodes.add(s)
        for t in dd.keys():
            nodes.add(t)
    names = sorted(list(nodes)) if include_nodes is None else list(include_nodes)
    idx = {n: i for i, n in enumerate(names)}
    N = len(names)
    W = np.zeros((N, N), dtype=np.float32)
    for s, dd in edge_map.items():
        if s not in idx:
            continue
        i = idx[s]
        for t, w in dd.items():
            if t not in idx:
                continue
            j = idx[t]
            if i == j:
                continue  # drop self-loops
            W[i, j] += float(w)
    return W, names

def infer_ei_labels(W: np.ndarray, names: List[str], zero_tol: float = 1e-6) -> np.ndarray:
    """
    +1 if sum(outgoing weights) > 0, -1 if < 0, else 0.
    """
    out_sum = W.sum(axis=1)
    ei = np.zeros(W.shape[0], dtype=np.float32)
    ei[out_sum > zero_tol] = +1.0
    ei[out_sum < -zero_tol] = -1.0
    return ei

# ------------- main  -------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", required=True, help="Path to CElegansNeuronTables.xls (or .xlsx)")
    ap.add_argument("--out", default="ce", help="Output prefix (default: ce)")
    ap.add_argument("--include-muscles", action="store_true", help="Include neuron→muscle edges/nodes")
    ap.add_argument("--engine", default=None, help='pandas Excel engine (auto; use "xlrd" for .xls or "openpyxl" for .xlsx)')
    ap.add_argument("--sr", type=float, default=0.99, help="Target spectral radius for scaled matrix")
    args = ap.parse_args()

    # Read sheets
    print(f"Reading Excel: {args.xls}")
    conn_df = read_sheet(args.xls, "Connectome", engine=args.engine)
    ntm_df  = read_sheet(args.xls, "NeuronsToMuscle", engine=args.engine)

    # Build edge maps
    print("Processing sheets…")
    conn_map = process_connectome(conn_df)
    ntm_map  = process_neuron_to_muscle(ntm_df)

    if args.include_muscles:
        combined = merge_edge_maps(conn_map, ntm_map)
    else:
        combined = conn_map

    # Build adjacency and names
    print("Building adjacency…")
    W, names = build_matrix(combined)

    # Infer EI labels from outgoing sign
    print("Inferring Dale (E/I) labels…")
    ei = infer_ei_labels(W, names)

    # Save natural matrix
    np.save(f"{args.out}_adj.npy", W.astype(np.float32))
    np.save(f"{args.out}_ei.npy", ei.astype(np.float32))
    with open(f"{args.out}_nodes.txt", "w") as f:
        for n in names:
            f.write(n + "\n")
    print(f"Saved: {args.out}_adj.npy  {args.out}_ei.npy  {args.out}_nodes.txt")

    # Save SR=target-scaled version
    print(f"Scaling to spectral radius ≈ {args.sr} …")
    W_scaled = scale_to_sr(W.astype(np.float64), target=args.sr).astype(np.float32)
    np.save(f"{args.out}_adj_sr{str(args.sr).replace('.','')}.npy", W_scaled)
    print(f"Saved: {args.out}_adj_sr{str(args.sr).replace('.','')}.npy")

    # Quick stats
    sr_nat = spectral_radius_numpy(W.astype(np.float64))
    sr_scl = spectral_radius_numpy(W_scaled.astype(np.float64))
    nnz = int((np.abs(W) > 0).sum())
    print(f"Nodes: {W.shape[0]} | Edges: {nnz} | SR(natural)={sr_nat:.3f} | SR(scaled)={sr_scl:.3f}")
    print("Done.")

if __name__ == "__main__":
    main()
