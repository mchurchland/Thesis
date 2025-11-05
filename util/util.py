import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import os

def imsave_heatmap(data: np.ndarray, row_labels, col_labels, title: str, fname: str):
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
        npy_path = os.path.join(base_dir, "ce_names.npy")
        txt_path = os.path.join(base_dir, "ce_names.txt")
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
def run_reservoir(W: Tensor, Win: Tensor, u: Tensor, leak: float) -> Tensor:
    """
    u: [T, 1] input sequence on DEVICE
    Returns states X: [T, N]
    """
    from heatmap import DEVICE
    N = W.shape[0]
    T = u.shape[0]
    z = torch.zeros(N, device=DEVICE)
    X = torch.zeros(T, N, device=DEVICE)
    for t in range(T):
        h = torch.tanh(W @ z + (Win @ u[t:t+1, :].T).squeeze())
        z = (1 - leak) * z + leak * h
        X[t] = z
    return X

def build_reservoir(
    feature_conn: str,         # 'cel', 'deg_shuffle', 'ws_p=1.0', 'ws_p=0.1', 'ws_p=0.0', 'er_p=...'
    feature_weights: str,      # 'bio', 'rand_disc', 'rand_gauss'
    feature_dale: str,         # 'none', 'e80i20', 'ei_from_cel'
    target_sr: float | None,   # <--- scale by spectral radius to this (None = unchanged)
    N: int,
    ce_W_bio: np.ndarray | None,
    ce_ei: np.ndarray | None,
    ws_k: int,
    input_scale: float,
    seed: int,
    drive_idx: np.ndarray | None = None,   # targeted drive for CEL rows if desired
    nnz_target: int | None = None,         # <--- desired number of edges (from CE)
) -> tuple[Tensor, Tensor, Tensor | None, float, float]:
    """
    Returns:
      Wt, Win, ei_t, rho_nat, rho_post
    """
    from heatmap import DEVICE
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # ---------- base adjacency/weights ----------
    if feature_conn == "cel":
        if ce_W_bio is None:
            # fallback WS mask then fill weights ~N(0,1)
            A = ws_adjacency(N, ws_k, 0.1, rng).astype(np.float32)
            mask = (A != 0).astype(np.float32)
            if nnz_target is not None:
                mask = _match_edge_count(mask.astype(bool), nnz_target, rng).astype(np.float32)
            W = mask * rng.normal(0.0, 1.0, size=(N, N)).astype(np.float32)
            ei_t = None
        else:
            W = ce_W_bio.copy().astype(np.float32)
            N = W.shape[0]
            # Keep the CE edge set as-is; if a different nnz_target was provided, ignore for CEL row.
            # Row-normalize magnitudes for stability like before:
            row_abs = np.sum(np.abs(W), axis=1, keepdims=True) + 1e-8
            W = W / row_abs
            ei_t = torch.from_numpy(ce_ei) if ce_ei is not None else None

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
        # row-normalize to avoid huge rows
        row_abs = np.sum(np.abs(W), axis=1, keepdims=True) + 1e-8
        W = W / row_abs
        ei_t = torch.from_numpy(ce_ei) if ce_ei is not None else None
        N = W.shape[0]

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
        ei_t = None

    else:
        raise ValueError(f"Unknown feature_conn: {feature_conn}")

    # Weight scheme overrides
    if feature_weights == "rand_disc":
        signs = rng.choice([-1.0, 1.0], size=W.shape).astype(np.float32)
        W = (np.abs(W) > 0).astype(np.float32) * signs
    elif feature_weights == "rand_gauss":
        mags = rng.normal(0.0, 1.0, size=W.shape).astype(np.float32)
        W = (np.abs(W) > 0).astype(np.float32) * mags
    elif feature_weights == "bio":
        if ce_W_bio is None and feature_conn != "cel":
            # emulate heavy-tail signed weights if no CE magnitudes
            mask = (np.abs(W) > 0).astype(np.float32)
            m = int(mask.sum())
            mags  = rng.lognormal(mean=-1.0, sigma=0.5, size=m).astype(np.float32)
            signs = rng.choice([-1.0, 1.0], size=m).astype(np.float32)
            W2 = np.zeros_like(W, dtype=np.float32)
            W2[mask != 0] = mags * signs
            W = W2
        # else: 'cel' with CE weights already prepared

    # Torchify
    Wt = torch.from_numpy(W).to(DEVICE)

    # Dale (optional)
    if feature_dale != "none":
        if feature_dale == "ei_from_cel" and ce_ei is not None:
            signs = torch.from_numpy(ce_ei).to(DEVICE)
        else:
            rng_np = np.random.default_rng(seed)
            n_exc = int(round(0.8 * N))
            signs_np = np.ones(N, dtype=np.float32)
            signs_np[n_exc:] = -1.0
            rng_np.shuffle(signs_np)
            signs = torch.from_numpy(signs_np).to(DEVICE)
        # Enforce signs by zeroing inconsistent outgoing signs
        Wpos = torch.clamp(Wt, min=0)
        Wneg = torch.clamp(Wt, max=0)
        mask_exc = (signs > 0).view(-1, 1)
        mask_inh = ~mask_exc
        Wt = torch.zeros_like(Wt)
        Wt[mask_exc.expand_as(Wt)] = Wpos[mask_exc.expand_as(Wpos)]
        Wt[mask_inh.expand_as(Wt)] = Wneg[mask_inh.expand_as(Wneg)]
        Wt.fill_diagonal_(0.0)

    # --- scale by spectral radius (this is the requested change) ---
    rho_nat = spectral_radius_power(Wt)
    Wt = scale_to_sr(Wt, target_sr)
    rho_post = spectral_radius_power(Wt)

    # --- Input weights Win ---
    if drive_idx is not None and len(drive_idx) > 0:
        Win = torch.zeros(Wt.shape[0], 1, device=DEVICE)
        Win[torch.as_tensor(drive_idx, device=DEVICE, dtype=torch.long), 0] = 1.0
        Win = Win * (input_scale / (Win.norm() + 1e-12))
    else:
        Win = torch.randn(Wt.shape[0], 1, device=DEVICE) * input_scale

    return Wt, Win, ei_t, rho_nat, rho_post

@torch.no_grad()
def spectral_radius_power(W: Tensor, iters: int = 200) -> float:
    """Power iteration estimate of spectral radius ρ(W)."""
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
    """Scale W so that spectral radius ρ(W) = target_sr; if None, return unchanged."""
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
    """
    n = mask.shape[0]
    M = mask.copy().astype(bool)
    np.fill_diagonal(M, False)
    current = int(M.sum())
    if current == target_m:
        return M.astype(np.float32)

    all_idx = [(i, j) for i in range(n) for j in range(n) if i != j]

    if current > target_m:
        # remove extras
        ones = np.argwhere(M)
        drop_k = current - target_m
        keep_sel = rng.choice(ones.shape[0], size=(ones.shape[0] - drop_k), replace=False)
        keep_pairs = ones[keep_sel]
        M[:] = False
        for i, j in keep_pairs:
            M[i, j] = True
    else:
        # add missing
        zeros = np.array([(i, j) for (i, j) in all_idx if not M[i, j]])
        add_k = target_m - current
        if add_k > 0 and zeros.size > 0:
            pick = rng.choice(zeros.shape[0], size=min(add_k, zeros.shape[0]), replace=False)
            for i, j in zeros[pick]:
                M[i, j] = True
            # if still short (zeros exhausted), leave as is
    np.fill_diagonal(M, False)
    return M.astype(np.float32)

def er_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Directed Erdős–Rényi; independent directions."""
    A = rng.random((n, n)) < p
    np.fill_diagonal(A, False)
    return A.astype(np.float32)

def ws_adjacency(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Undirected Watts–Strogatz adjacency (bool)."""
    assert k % 2 == 0 and k < n and 0.0 <= p <= 1.0
    A = np.zeros((n, n), dtype=bool)
    # ring
    for i in range(n):
        for j in range(1, k // 2 + 1):
            A[i, (i + j) % n] = True
            A[i, (i - j) % n] = True
    # rewire symmetrically
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                old = (i + j) % n
                if not A[i, old]:
                    continue
                A[i, old] = False; A[old, i] = False
                candidates = np.where(~A[i] & (np.arange(n) != i))[0]
                if len(candidates) == 0:
                    A[i, old] = True; A[old, i] = True
                else:
                    new = rng.choice(candidates)
                    A[i, new] = True; A[new, i] = True
    np.fill_diagonal(A, False)
    return A.astype(np.float32)


def degree_matched_shuffle_directed(A: np.ndarray, tries: int = 10_000,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    A = A.copy().astype(bool)
    n = A.shape[0]
    np.fill_diagonal(A, False)
    edges = np.argwhere(A)
    m = edges.shape[0]
    if m < 2:
        return A.astype(np.float32)
    for _ in range(tries):
        idx = rng.choice(m, size=2, replace=False)
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]
        if len({a, b, c, d}) < 4:
            continue
        if a == d or c == b:
            continue
        if A[a, d] or A[c, b]:
            continue
        A[a, b] = False; A[c, d] = False
        A[a, d] = True;  A[c, b] = True
        edges[idx[0]] = [a, d]
        edges[idx[1]] = [c, b]
    return A.astype(np.float32)

def spectral_norm(W: Tensor) -> float:
    return float(torch.linalg.svdvals(W)[0])

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)