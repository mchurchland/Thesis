# build_ce_connectome.py
# ------------------------------------------------------------
# Parse CElegansNeuronTables.xls and produce:
#   ce_adj.npy        (float32 NxN adjacency with signed synapse counts)
#   ce_nodes.txt      (node names, one per line, same order as matrix)
#
# Sheets expected:
#   - "Connectome"         with columns: Origin, Target, Number of Connections, Neurotransmitter
#   - "NeuronsToMuscle"    with columns: Neuron, Muscle, Number of Connections, Neurotransmitter
#
# Notes
# - Edge sign rule: GABA → negative; ACh/Glutamate → positive; others default to 0 (changeable).
# - Duplicate edges are summed.
# - Muscles can be included or dropped via flag.
#
# Usage:
#   python build_ce_connectome.py --xls CElegansNeuronTables.xls --out ce \
#       --include-muscles  # drop this flag to exclude muscles
#   python build_ce_connectome.py --xls new_cel.xls --new-cel --out ce
#
# Requirements:
#   pip/conda: pandas numpy xlrd (for .xls)  OR openpyxl (for .xlsx with engine="openpyxl")
# ------------------------------------------------------------
import argparse
import numpy as np
import pandas as pd
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
    if s in {"Acetylcholine_Tyramine"}:
        return "TYR"
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
    GABA -> -1; ACh/GLU -> +1; others → 0 by default (set default_pos_if_unknown=False to make 0).
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
            print(row)
            continue
        weight = float(row[num])
        sign = nt_to_edge_sign(row[nt])
        add_edge(acc, src, dst, sign * weight)
    return acc

def _new_cel_sign_token(sign_value) -> str:
    if pd.isna(sign_value):
        return ""
    return str(sign_value).strip().lower()

def new_cel_sign_is_unknown(sign_value) -> bool:
    """Return True for new_cel.xls Sign values with no usable prediction."""
    s = _new_cel_sign_token(sign_value)
    return s in {
        "",
        "complex",
        "no pred",
        "no_pred",
        "nopred",
        "no prediction",
        "unknown",
        "unk",
        "0",
        "zero",
    }

def new_cel_sign_to_edge_sign(sign_value) -> int:
    """Map new_cel.xls Sign values to known edge signs; unknown/complex -> 0."""
    s = _new_cel_sign_token(sign_value)
    if s in {"+", "plus", "pos", "positive"}:
        return +1
    if s in {"-", "minus", "neg", "negative"}:
        return -1
    if new_cel_sign_is_unknown(sign_value):
        return 0
    raise ValueError(f"Unknown new_cel Sign value: {sign_value!r}")

def process_new_cel(
    df: pd.DataFrame,
    *,
    return_unknown: bool = False,
) -> Dict[str, Dict[str, float]] | Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Expect new_cel.xls Table Export columns: Source, Target, Edge Weight, Sign.
    Sign values complex/no pred are kept as zero-weight edges in the signed
    adjacency. If return_unknown=True, also return an edge map containing their
    unsigned magnitudes for runtime random sign assignment.
    """
    cols = {c.lower().strip(): c for c in df.columns}
    source = cols.get("source")
    target = cols.get("target")
    weight_col = cols.get("edge weight") or cols.get("edge_weight")
    sign_col = cols.get("sign")

    if not all([source, target, weight_col, sign_col]):
        raise ValueError("new_cel sheet must have columns: Source, Target, Edge Weight, Sign")

    acc: Dict[str, Dict[str, float]] = {}
    unknown_acc: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        src = str(row[source]).strip()
        dst = str(row[target]).strip()
        if src == "" or dst == "" or pd.isna(row[weight_col]):
            print(row)
            continue
        weight = float(row[weight_col])
        sign = new_cel_sign_to_edge_sign(row[sign_col])
        if sign == 0:
            add_edge(unknown_acc, src, dst, abs(weight))
        add_edge(acc, src, dst, sign * weight)
    if return_unknown:
        return acc, unknown_acc
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

# ------------- main  -------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", required=True, help="Path to CElegansNeuronTables.xls (or .xlsx)")
    ap.add_argument("--out", default="ce", help="Output prefix (default: ce)")
    ap.add_argument("--include-muscles", action="store_true", help="Include neuron→muscle edges/nodes",default=False)
    ap.add_argument("--engine", default=None, help='pandas Excel engine (auto; use "xlrd" for .xls or "openpyxl" for .xlsx)')
    ap.add_argument("--new-cel", action="store_true", help='Read the new_cel.xls "Table Export" format')
    args = ap.parse_args()

    # Read sheets
    print(f"Reading Excel: {args.xls}")
    print("Processing sheets…")
    if args.new_cel:
        conn_df = read_sheet(args.xls, "Table Export", engine=args.engine)
        combined, unknown_combined = process_new_cel(conn_df, return_unknown=True)
        print(len(set(conn_df["Source"].to_numpy()).union(set(conn_df["Target"].to_numpy()))))
    else:
        conn_df = read_sheet(args.xls, "Connectome", engine=args.engine)
        ntm_df  = read_sheet(args.xls, "NeuronsToMuscle", engine=args.engine)
        print(len(set(conn_df["Origin"].to_numpy()).union((set(conn_df["Target"].to_numpy())))))

        #print(conn_df["Origin"],conn_df['Target'])
        # Build edge maps
        conn_map = process_connectome(conn_df)
        ntm_map  = process_neuron_to_muscle(ntm_df)

        if args.include_muscles:
            combined = merge_edge_maps(conn_map, ntm_map)
        else:
            combined = conn_map
        unknown_combined = None

    # Build adjacency and names
    print("Building adjacency…")
    W, names = build_matrix(combined)
    W_unknown = None
    if unknown_combined is not None:
        W_unknown, _ = build_matrix(unknown_combined, include_nodes=names)
    num_gt, num_lt = (W>0).sum(),(W<0).sum()
    p_neg  = num_lt/(num_gt+num_lt)
    print(f"P(swap) = {2*p_neg*(1-p_neg)}")
    print(f"E(swap) = {2*p_neg*(1-p_neg)*(num_gt+num_lt)}")

    # Save natural matrix
    np.save(f"{args.out}_adj.npy", W.astype(np.float32))
    if W_unknown is not None:
        np.save(f"{args.out}_unknown_sign_weights.npy", np.abs(W_unknown).astype(np.float32))
    with open(f"{args.out}_nodes.txt", "w") as f:
        for n in names:
            f.write(n + "\n")
    saved = f"Saved: {args.out}_adj.npy  {args.out}_nodes.txt"
    if W_unknown is not None:
        saved += f"  {args.out}_unknown_sign_weights.npy"
    print(saved)

    # Save SR=target-scaled version


    nnz = int((np.abs(W) > 0).sum())
    unknown_nnz = int((np.abs(W_unknown) > 0).sum()) if W_unknown is not None else 0
    print(f"Nodes: {W.shape[0]} | Edges: {nnz}")
    if W_unknown is not None:
        print(f"Unknown/complex sign edges available for random assignment: {unknown_nnz}")
    print("Done.")

if __name__ == "__main__":
    main()
