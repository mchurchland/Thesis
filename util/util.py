import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import os
import networkx as nx

def imsave_heatmap(data: np.ndarray, row_labels, col_labels, title: str, fname: str):
    """Save a labeled heatmap using Matplotlib (Hunter, 2007, Comput. Sci. Eng. 9:90-95). See: https://github.com/matplotlib/matplotlib/blob/main/examples/images_contours_and_fields/image_annotated_heatmap.py"""
    plt.figure(figsize=(1.6 + 1.1*len(col_labels), 1.6 + 0.9*len(row_labels)))
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    im = plt.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(im)
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig("Results/"+fname, dpi=140)
    plt.close()
    print(f"Saved {fname}")


def load_connectome(adj_path: str | None, ei_path: str | None):
    """
    Load C. elegans adjacency and EI labels (Varshney et al., 2011, PLoS Comput. Biol. 7:e1001066).
    See: https://github.com/OpenWorm/CElegansNeuroML/blob/master/CElegansNeuronTables.xlsx

    Returns:
      W_bio: np.ndarray [N,N] or None
      ei_labels: np.ndarray [N] with values in {-1,0,+1} or None
      name2idx: dict[str,int] mapping neuron name -> index, or None

    Behavior:
      - Replaces NaN/inf in W with 0, zeros self-loops, dtype float32.
      - EI kept in {-1,0,+1}; tiny values -> 0; dtype float32.
      - If a names file exists alongside the adjacency, builds name2idx.
        Looks for 'ce_names.npy' (array of str) or 'ce_names.txt' (one per line).
    """
    W_bio, ei_labels, name2idx = None, None, None

    # ---- adjacency ----
    if adj_path is not None and os.path.isfile(adj_path):
        W_bio = np.load(adj_path)
        if W_bio.ndim != 2 or W_bio.shape[0] != W_bio.shape[1]:
            raise ValueError("CE adjacency must be a square 2D array.")
        W_bio = W_bio.astype(np.float32, copy=False)

        # clean numerics & remove self-loops
        if not np.isfinite(W_bio).all():
            W_bio = np.where(np.isfinite(W_bio), W_bio, 0.0).astype(np.float32, copy=False)
        np.fill_diagonal(W_bio, 0.0)

        # try to load names from same folder as adj
        base_dir = os.path.dirname(adj_path)
        names = None
        npy_path = os.path.join(base_dir, "ce_nodes.npy")
        txt_path = os.path.join(base_dir, "ce_nodes.txt")
        if os.path.isfile(npy_path):
            names = np.load(npy_path)
        elif os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                names = np.array([ln.strip() for ln in f if ln.strip()])

        if names is not None:
            if len(names) != W_bio.shape[0]:
                raise ValueError("Names length must equal adjacency size.")
            # build mapping
            name2idx = {str(n): i for i, n in enumerate(names)}

    # ---- EI labels ----
    if ei_path is not None and os.path.isfile(ei_path):
        ei_labels = np.load(ei_path).astype(np.float32, copy=False)
        if ei_labels.ndim != 1:
            raise ValueError("EI labels must be a 1D array.")
        if W_bio is not None and ei_labels.shape[0] != W_bio.shape[0]:
            raise ValueError("EI labels length must match adjacency size.")
        # sanitize: tiny -> 0, then sign to {-1,0,+1}
        ei_clean = ei_labels.copy()
        ei_clean[np.abs(ei_clean) < 1e-6] = 0.0
        ei_clean = np.sign(ei_clean).astype(np.float32, copy=False)
        ei_labels = ei_clean

    return W_bio, ei_labels, name2idx



@torch.no_grad()
def run_reservoir(W: torch.Tensor,
                  Win: torch.Tensor,
                  u: torch.Tensor,
                  leak: float) -> torch.Tensor:
    """
    Echo state update for a single-reservoir run (Jaeger, 2002, GMD Report 152).
    See: https://github.com/cknd/pyESN/blob/master/pyESN.py#L32

    W:   [N, N]
    Win: [N, 1] or [N]
    u:   [T, 1] or [T]
    Returns X: [T, N]
    """
    device = W.device
    N = W.shape[0]
    T = u.shape[0]

    # normalize shapes once
    u_flat = u.view(T)                    # [T]
    win_vec = Win.view(N)                 # [N]

    z = torch.zeros(N, device=device)
    X = torch.empty(T, N, device=device)  # no need to zero

    one_minus_leak = 1.0 - leak

    for t in range(T):
        # h = tanh(W z + win * u_t)
        # use addmv: y + A x  (all vectors) for better BLAS path
        h = torch.tanh(torch.addmv(win_vec * u_flat[t], W, z))
        # z = (1 - leak) * z + leak * h, in-place
        z.mul_(one_minus_leak).add_(h, alpha=leak)
        X[t].copy_(z)

    return X

def build_reservoir(
    target_sr: float | None,   # <--- scale by spectral radius to this (None = unchanged)
    ce_ei: np.ndarray | None,
    input_scale: float,
    seed: int,
    N: int| None = None,
    ws_k: int | None = None,
    ce_W_bio: np.ndarray | None = None,
    feature_conn: str | None = None,         # 'cel', 'deg_shuffle', 'ws_p=1.0', 'ws_p=0.1', 'ws_p=0.0', 'er_p=...'
    feature_weights: str | None = None,      # 'bio', 'rand_disc', 'rand_gauss'
    drive_idx: np.ndarray | None = None,   # targeted drive for CEL rows if desired
    nnz_target: int | None = None,         # <--- desired number of edges (from CE)
    DEVICE: torch.device | None = None,
    per_neg: float| None = None
) -> tuple[Tensor, Tensor, Tensor, float]:
    """
    Construct a reservoir with selectable topology/weights: CEL/degree-shuffle (Milo et al., 2002, Science 298:824-827),
    ER (Erdos & Renyi, 1959, Publ. Math. Debrecen 6:290-297), WS (Watts & Strogatz, 1998, Nature 393:440-442),
    and spectral-radius scaling for echo state networks (Jaeger, 2002, GMD Report 152); Dale sign constraints follow
    standard excitatory/inhibitory segregation (Eccles, 1964, The Physiology of Synapses).
    See: https://github.com/networkx/networkx/blob/main/networkx/generators/random_graphs.py (ER),
         https://github.com/networkx/networkx/blob/main/networkx/generators/smallworld.py (WS),
         https://github.com/networkx/networkx/blob/main/networkx/algorithms/swap.py (degree-preserving swaps),
         https://github.com/cknd/pyESN/blob/master/pyESN.py (spectral-radius scaling / ESN setup).

    Returns:
      Wt, Win, ei_t, rho_nat, rho_post
    """
    #print(target_sr,ce_ei,input_scale,seed,N,ws_k,ce_W_bio,feature_conn,feature_weights,drive_idx,nnz_target,DEVICE,Normalize,per_neg)
    #quit()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ---------- base adjacency/weights ----------
    if feature_conn == "cel":
        if ce_W_bio is None:
            raise RuntimeError("you must pass in bio weights (ce_W_bio) to use the celegan reservoirs")
        else:
            W = ce_W_bio.copy().astype(np.float32)
            # Keep the CE edge set as-is; if a different nnz_target was provided, ignore for CEL row.
            # Row-normalize magnitudes for stability like before:
            ei_t = None

    elif feature_conn == 'local_sign':
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        sel_p = W > 0 ## get the positive weights of the selection
        sel_n = W < 0 ## get the negative weights of the selection


        # match cel+randN: N(0, 1) on existing edges
        num_pos = int(sel_p.sum())
        num_neg = int(sel_n.sum())
        if num_pos:
            W[sel_p] = np.abs(rng.normal(loc=0.0, scale=1.0, size=num_pos).astype(np.float32))
        if num_neg:
            W[sel_n] = -np.abs(rng.normal(loc=0.0, scale=1.0, size=num_neg).astype(np.float32))
        ei_t = None
    
    elif feature_conn == 'local_sign+flat':
        #from reservoir_variants import _shuffle_ce_weights_except_1
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        sel_p = W > 0 ## get the positive weights of the selection
        sel_n = W < 0 ## get the negative weights of the selection

        #W = _shuffle_ce_weights_except_1(Wbio = W,rng=rng)
        mags = rng.uniform(0.0, 1.0, size=W.shape).astype(np.float32)
        W = (np.abs(W) > 0).astype(np.float32) * mags
        W[sel_p] = np.abs(W[sel_p])
        W[sel_n] = -np.abs(W[sel_n])
        ei_t = None
    
    elif feature_conn == "local_sign+sample":
        from reservoir_variants import _sample_from_cel_sign
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        W = _sample_from_cel_sign(Wbio = W,rng=rng)

        ei_t = None

    elif feature_conn == "local_sign+binary":
        from reservoir_variants import _cel_to_bin
        if ce_W_bio is None:
            raise ValueError("Local sign match requires CE adjacency.")
        W = ce_W_bio.copy().astype(np.float32)

        sel_p = W > 0 ## get the positive weights of the selection
        sel_n = W < 0 ## get the negative weights of the selection

        W = _cel_to_bin(Wbio = W)
        W[sel_p] = np.abs(W[sel_p])
        W[sel_n] = -np.abs(W[sel_n])
        ei_t = None
    
    
    elif feature_conn == "deg_shuffle":
        if ce_W_bio is None:
            raise ValueError("Degree-matched shuffle requires CE adjacency.")
        A = (ce_W_bio != 0).astype(np.float32)
        As = degree_matched_shuffle_directed(A, tries=20_000, rng=rng)
        mask = (As != 0).astype(np.float32)
        if nnz_target is not None:
            mask = _match_edge_count(mask.astype(bool), nnz_target, rng).astype(np.float32)
        if feature_weights == "bio":
            vals = ce_W_bio[ce_W_bio != 0].astype(np.float32)
            rng.shuffle(vals)
            W = np.zeros_like(mask, dtype=np.float32)
            W[mask != 0] = vals[: int(mask.sum())]
        else:
            W = mask * rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
        ei_t = torch.from_numpy(ce_ei) if ce_ei is not None else None

    elif feature_conn.startswith("ws_p="):
        p = float(feature_conn.split("=")[1])
        A = ws_adjacency(N, ws_k, p, rng).astype(np.float32)
        mask = (A != 0).astype(np.float32)
        if nnz_target is not None:
            mask = _match_edge_count(mask.astype(bool), nnz_target, rng).astype(np.float32)
        W = mask * rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
        ei_t = None

    elif feature_conn.startswith("er_p="):
        p = float(feature_conn.split("=")[1])
        A = er_adjacency(N, p, rng).astype(np.float32)
        mask = (A != 0).astype(np.float32)
        if nnz_target is not None:
            mask = _match_edge_count(mask.astype(bool), nnz_target, rng).astype(np.float32)
        W = mask * rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
        if per_neg:
            nz = np.nonzero(W)
            n_neg = int(per_neg * len(nz[0]))
            idx = np.arange(len(nz[0]))
            rng.shuffle(idx)
            sel = (nz[0][idx[:n_neg]], nz[1][idx[:n_neg]])
            W = np.abs(W)
            W[sel] = -1*W[sel]
        ei_t = None
        
    else:
        W = ce_W_bio.copy().astype(np.float32)
        ei_t = None

    # Weight scheme overrides
    if feature_weights == "rand_disc":
        signs = rng.choice([-1.0, 1.0], size=W.shape).astype(np.float32)
        W = (np.abs(W) > 0).astype(np.float32) * signs
    elif feature_weights == "rand_gauss":
        mags = rng.normal(0.0, 1.0, size=W.shape).astype(np.float32)
        W = (np.abs(W) > 0).astype(np.float32) * mags
        
        # else: 'cel' with CE weights already prepared

    # Torchify
    

    # Apply Dale's Law from ce_ei 
    if ce_ei is not None:
        diag = np.diag(ce_ei).astype(np.float32)
        W  = np.matmul(diag,np.abs(W)).astype(np.float32)

    Wt = torch.from_numpy(W).to(DEVICE)
    # --- scale by spectral radius (this is the requested change) ---
    rho_nat = spectral_radius_power(Wt)
    Wt = scale_to_sr(Wt, target_sr)
    #rho_post = spectral_radius_power(Wt)

    # --- Input weights Win ---
    if drive_idx is not None and len(drive_idx) > 0:
        Win = torch.zeros(Wt.shape[0], 1, device=DEVICE)
        # Index tensors must be integer type.
        Win[torch.as_tensor(drive_idx, device=DEVICE, dtype=torch.long), 0] = 1.0
        Win = Win * (input_scale / (Win.norm() + 1e-12))
    else:
        Win = torch.randn(Wt.shape[0], 1, device=DEVICE, dtype=Wt.dtype) * input_scale

    return Wt, Win, ei_t, rho_nat

@torch.no_grad()
def spectral_radius_power(W: Tensor, iters: int = 200) -> float:
    """
    Spectral radius via torch.linalg.eigvals with a power-iteration fallback
    (Golub & Van Loan, 2013, Matrix Computations 4th ed.).
    """
    try:
        eigs = torch.linalg.eigvals(W)
        return float(torch.max(torch.abs(eigs)).item())
    except Exception:
        n = W.shape[0]
        v = torch.randn(n, device=W.device)
        v = v / (v.norm() + 1e-12)
        lam = 0.0
        for _ in range(iters):
            v = W @ v
            nrm = v.norm()
            if float(nrm) < 1e-12:
                break
            v = v / nrm
            lam = float((v @ (W @ v)) / (v @ v + 1e-12))
        return abs(lam)

@torch.no_grad()
def scale_to_sr(W: torch.Tensor, target_sr: float | None):
    """
    Scale W so that spectral radius rho(W) = target_sr (Jaeger, 2002, GMD Report 152); if None, return unchanged.
    See: https://github.com/cknd/pyESN/blob/master/pyESN.py#L59 (spectral-radius scaling in ESNs).
    """
    if target_sr is None:
        return W
    sr = spectral_radius_power(W)
    if sr < 1e-9:
        return W
    return (target_sr / sr) * W

def _match_edge_count(mask: np.ndarray, target_m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Given a boolean mask (no self-loops), randomly add/remove edges to match target_m nnz.
    Returns a boolean mask with exactly target_m ones (and zero diagonal).

    Random edge add/drop to match density (Newman, 2010, Networks: An Introduction).
    Uses NetworkX DiGraph utilities; see https://github.com/networkx/networkx/blob/main/networkx/classes/digraph.py
    """
    G = nx.from_numpy_array(mask.astype(bool), create_using=nx.DiGraph)
    G.remove_edges_from(nx.selfloop_edges(G))
    current = G.number_of_edges()
    nodes = list(G.nodes())

    if current > target_m:
        edges = list(G.edges())
        drop_idx = rng.choice(len(edges), size=current - target_m, replace=False)
        G.remove_edges_from([edges[i] for i in drop_idx])
    elif current < target_m:
        candidates = [(u, v) for u in nodes for v in nodes if u != v and not G.has_edge(u, v)]
        need = target_m - current
        if need > 0 and candidates:
            add_idx = rng.choice(len(candidates), size=min(need, len(candidates)), replace=False)
            G.add_edges_from([candidates[i] for i in add_idx])

    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.to_numpy_array(G, dtype=np.float32)

def er_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Directed Erdos-Renyi graph with independent directions (Erdos & Renyi, 1959, Publ. Math. Debrecen 6:290-297).
    Uses NetworkX gnp_random_graph with directed=True; see https://github.com/networkx/networkx/blob/main/networkx/generators/random_graphs.py#L310
    """
    G = nx.gnp_random_graph(n, p, seed=rng, directed=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.to_numpy_array(G, dtype=np.float32)

def ws_adjacency(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Undirected Watts-Strogatz small-world adjacency (Watts & Strogatz, 1998, Nature 393:440-442).
    Uses NetworkX watts_strogatz_graph; see https://github.com/networkx/networkx/blob/main/networkx/generators/smallworld.py#L14
    """
    assert k % 2 == 0 and k < n and 0.0 <= p <= 1.0
    G = nx.watts_strogatz_graph(n, k, p, seed=rng)
    A = nx.to_numpy_array(G, dtype=np.float32)
    np.fill_diagonal(A, 0.0)
    return A


def degree_matched_shuffle_directed(A: np.ndarray, tries: int,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Degree-preserving double-edge swap randomization for directed graphs
    (Milo et al., 2002, Science 298:824-827).
    """
    def can_swap(a,b,c,d):
        if (len({a,b,c,d}) < 4) or ( A[a,d] or A[c,b]): ## makes sure all the values are unique, if the connection between a and d already exists or c and b
            return False
        else:
            return True
    def edge_swap(a: int, b: int, c: int, d: int):
        edge1_weight = A[a,b].copy()
        edge2_weight = A[c,d].copy()
        A[a,b] = 0  # remove edge 1
        A[c,d] = 0  # remove edge 2
        A[a,d] = edge1_weight  # add edge 1 to new nodes
        A[c,b] = edge2_weight  # add edge 2 to new nodes
    if rng is None:
        raise ValueError("Need to pass in a random number generator")
    A = A.copy() ## make a copy of A
    np.fill_diagonal(A, False)
    edges = np.argwhere(A!=0)
    m = edges.shape[0]
    if m < 2:
        raise ValueError(f"Not enough edges to perform randomization. Found {m} edges. make sure the matrix is correct")

    retries = 0
    
    idx = rng.permutation(m)  
    #while not(idx.isempty) and retries < max_retries:
    i=0
    
    while i + 1 < len(idx) and retries < tries:
        a, b = edges[idx[i]] ## edge 1
        c, d = edges[idx[i+1]] ## edge 2
        if can_swap(a,b,c,d):
            edge_swap(a,b,c,d) ## swap the edges
            edges[idx[i]] = [a,d]
            edges[idx[i+1]] = [c,b]
            i+=2
            
        else:
            retries += 1
            rem = idx[i:]
            rng.shuffle(rem) ## shuffle the indices
            idx[i:] = rem ## replace the indices with the shuffled one
    return A.astype(np.float32)

def flip_percent(Wbio:np.ndarray,per:np.float32,rng:np.random.Generator):
    assert per >0 and per < 1
    W = Wbio.copy().astype(np.float32)
    nz = np.nonzero(W)
    num_to_flip = int(len(nz[0])*per)
    ind_to_flip  = rng.choice(nz,num_to_flip,replace=False)
    W[ind_to_flip] = -W[ind_to_flip]
    return W



def spectral_norm(W: Tensor) -> float:
    """Spectral norm via leading singular value (Golub & Van Loan, 2013, Matrix Computations 4th ed.). See: https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py"""
    return float(torch.linalg.svdvals(W)[0])

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

