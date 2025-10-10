# mc_connectome_video.py
# ---------------------------------------------------------------
# EXACT MC-test dynamics & sampling, rendered as a video.
# Two viewers:
#   • Middle: BIO model (any heatmap row/col you pick)
#   • Right : STANDARD ESN (Gaussian weights, ER sparsity, SR=policy)
#
# Matches your reservoir_diagnostics_heatmaps.py for:
#   - u_t ~ Uniform[0,1] with global SEED
#   - Win ~ N(0,1)*INPUT_SCALE (per-model seed rule)
#   - Same tanh+leak update, same SR policy per selected column
# ---------------------------------------------------------------

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import torch
from torch import Tensor
from moviepy import ImageSequenceClip

from connectome_graph import ConnectomeViewer

# =================== Paths & constants (match MC script) ===================

CE_ADJ_PATH = "ce_adj.npy"
CE_EI_PATH  = "ce_ei.npy"

N_DEFAULT = 400

# If you want the full MC window, set these to your diagnostics values.
WASHOUT = 50
T_TRAIN = 0
T_TEST  = 0
SEED    = 0

WS_K = 40
INPUT_SCALE = 0.7
LEAK = 0.2

SR_TARGET_OPT = 0.99
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# =================== Video / viz config ===================

OUT_PATH = "mc_connectome.mp4"
FPS = 25
RES_SCALE = 2
TITLE_TEXT = "Bio vs Std ESN (Exact MC drive)"
VMIN_PCT = 1.0
VMAX_PCT = 99.0
NODE_SIZE = 46

# =================== Feature grid (bio model parity) ===================

ROW_DEFS = [
    ("CEL + bioW",             dict(conn="cel",         weights="bio")),
    ("CEL + randN",            dict(conn="cel",         weights="rand_gauss")),
    ("CEL-shuf + randN",       dict(conn="deg_shuffle", weights="rand_gauss")),
    ("WS p=1.0 + randN",       dict(conn="ws_p=1.0",    weights="rand_gauss")),
    ("WS p=0.1 + randN",       dict(conn="ws_p=0.1",    weights="rand_gauss")),
    ("WS p=0.0 + randN",       dict(conn="ws_p=0.0",    weights="rand_gauss")),
    ("WS p=0.1 + randDisc",    dict(conn="ws_p=0.1",    weights="rand_disc")),
]

COL_DEFS = [
    ("no-Dale, SR=natural",    dict(dale="none",       sr="natural")),
    ("no-Dale, SR=0.99",       dict(dale="none",       sr="target")),
    ("Dale 80:20, SR=0.99",    dict(dale="e80i20",     sr="target")),
    ("Dale from CEL, SR=0.99", dict(dale="ei_from_cel",sr="target")),
    ("Dale 80:20, SR=natural", dict(dale="e80i20",     sr="natural")),
]

# Select the BIO heatmap cell to reproduce
RI = 0  # 0..len(ROW_DEFS)-1
CI = 1  # 0..len(COL_DEFS)-1

# STANDARD ESN params (right panel)
ESN_SPARSITY = 0.1           # connection prob (ER graph)
ESN_WEIGHT_STD = 1.0         # raw Gaussian std, then SR scaled
ESN_APPLY_DALE = False       # standard ESN ignores Dale’s law

# =================== Utilities (parity with MC script) ===================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_connectome(adj_path: str | None, ei_path: str | None):
    W_bio, ei_labels = None, None
    if adj_path is not None and os.path.isfile(adj_path):
        W_bio = np.load(adj_path).astype(np.float32, copy=False)
        assert W_bio.ndim == 2 and W_bio.shape[0] == W_bio.shape[1], "CE adjacency must be square."
    if ei_path is not None and os.path.isfile(ei_path):
        ei_labels = np.load(ei_path).astype(np.float32)
        ei_labels = np.where(ei_labels > 0, 1.0, -1.0).astype(np.float32)
        if W_bio is not None:
            assert ei_labels.shape[0] == W_bio.shape[0], "EI labels length must match adjacency size."
    return W_bio, ei_labels

def ws_adjacency(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    assert k % 2 == 0 and k < n and 0.0 <= p <= 1.0
    A = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            A[i, (i + j) % n] = True
            A[i, (i - j) % n] = True
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

def degree_matched_shuffle_directed(A: np.ndarray, tries: int = 10_000, rng: np.random.Generator | None = None) -> np.ndarray:
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

def spectral_radius(M: Tensor) -> float:
    with torch.no_grad():
        v = torch.randn(M.shape[1], device=M.device)
        v = v / (v.norm() + 1e-9)
        for _ in range(100):
            v = M @ v
            nrm = v.norm()
            if nrm < 1e-12:
                break
            v = v / nrm
        lam = (v @ (M @ v)) / (v @ v + 1e-12)
        return float(torch.abs(lam))

def scale_to_sr(W: Tensor, mode: str, sr_target: float | None = None) -> Tensor:
    if mode == "natural":
        return W
    assert sr_target is not None and sr_target > 0
    sr = spectral_radius(W)
    if sr < 1e-9:
        return W
    return (sr_target / sr) * W

def apply_dales_law(W: Tensor, dale_mode: str, ei_labels: Tensor | None,
                    ei_ratio: float = 0.8, rng: np.random.Generator | None = None) -> Tensor:
    if dale_mode == "none":
        return W
    N = W.shape[0]
    if dale_mode == "ei_from_cel" and ei_labels is not None:
        signs = ei_labels.to(W.device)
    else:
        if rng is None:
            rng = np.random.default_rng(SEED)
        n_exc = int(round(ei_ratio * N))
        signs_np = np.ones(N, dtype=np.float32)
        signs_np[n_exc:] = -1.0
        rng.shuffle(signs_np)
        signs = torch.from_numpy(signs_np).to(W.device)
    W_mag = torch.abs(W)
    W_signed = W_mag * signs.view(-1, 1)
    W_signed.fill_diagonal_(0.0)
    return W_signed

# ----- BIO model builder (parity with diagnostics) -----
def build_reservoir(
    feature_conn: str,
    feature_weights: str,
    feature_dale: str,
    feature_sr: str,
    N: int,
    ce_W_bio: np.ndarray | None,
    ce_ei: np.ndarray | None,
    ws_k: int,
    sr_target: float,
    input_scale: float,
    seed: int
) -> tuple[Tensor, Tensor, Tensor | None]:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    if feature_conn == "cel":
        if ce_W_bio is None:
            A = ws_adjacency(N, ws_k, 0.1, rng).astype(np.float32)
            W = A.copy()
            W[W != 0] = rng.uniform(0.1, 1.0, size=W[W != 0].shape)
            ce_ei_t = None
        else:
            W = ce_W_bio.copy()
            N = W.shape[0]
            ce_ei_t = torch.from_numpy(ce_ei) if ce_ei is not None else None
    elif feature_conn == "deg_shuffle":
        if ce_W_bio is None:
            raise ValueError("Degree-matched shuffle requires CE adjacency.")
        A = (ce_W_bio != 0).astype(np.float32)
        As = degree_matched_shuffle_directed(A, tries=20_000, rng=rng)
        if feature_weights == "bio":
            mags = np.abs(ce_W_bio[ce_W_bio != 0])
            rng.shuffle(mags)
            W = np.zeros_like(As)
            W[As != 0] = mags[: np.count_nonzero(As)]
        else:
            W = As.copy()
        ce_ei_t = torch.from_numpy(ce_ei) if ce_ei is not None else None
        N = W.shape[0]
    elif feature_conn.startswith("ws_p="):
        p = float(feature_conn.split("=")[1])
        A = ws_adjacency(N, ws_k, p, rng).astype(np.float32)
        W = A.copy()
        ce_ei_t = None
    else:
        raise ValueError(f"Unknown feature_conn: {feature_conn}")

    if feature_weights in ("rand_disc", "rand_gauss") or feature_conn.startswith("ws_p="):
        mask = (np.abs(W) > 0).astype(np.float32)
        if feature_weights == "rand_disc":
            vals = rng.choice([-1.0, 1.0], size=mask.shape).astype(np.float32)
            W = mask * vals
        elif feature_weights == "rand_gauss":
            vals = rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
            W = mask * vals
        else:
            vals = rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32)
            W = mask * vals

    Wt = torch.from_numpy(W).to(DEVICE)
    ei_t = torch.from_numpy(ce_ei).to(DEVICE) if (feature_dale == "ei_from_cel" and ce_ei is not None) else None
    Wt = apply_dales_law(Wt, feature_dale, ei_t, ei_ratio=0.8, rng=rng)
    Wt = scale_to_sr(Wt, "target" if feature_sr == "target" else "natural",
                     sr_target=(SR_TARGET_OPT if feature_sr == "target" else None))

    Win = torch.randn(Wt.shape[0], 1, device=DEVICE) * input_scale
    return Wt, Win, ei_t

# ----- STANDARD ESN builder (ER + Gaussian, optional SR target) -----
def build_standard_esn(
    N: int,
    sparsity: float,
    weight_std: float,
    feature_sr: str,           # 'natural' | 'target'
    sr_target: float,
    input_scale: float,
    seed: int
) -> tuple[Tensor, Tensor]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    # ER mask
    mask = (rng.random((N, N)) < sparsity).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    # Gaussian weights
    W = rng.normal(0.0, weight_std, size=(N, N)).astype(np.float32) * mask
    Wt = torch.from_numpy(W).to(DEVICE)
    # SR policy
    Wt = scale_to_sr(Wt, "target" if feature_sr == "target" else "natural",
                     sr_target=(sr_target if feature_sr == "target" else None))
    # Input weights
    Win = torch.randn(N, 1, device=DEVICE) * input_scale
    return Wt, Win

# =================== Rendering helpers ===================

def _mpl_fig_to_bgr(fig, scale: int = RES_SCALE):
    orig_dpi = fig.dpi
    fig.set_dpi(orig_dpi * scale)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    argb = np.frombuffer(fig.canvas.tostring_argb(), np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(argb[:, :, 1:], cv2.COLOR_RGB2BGR)
    fig.set_dpi(orig_dpi)
    return bgr

def _panel_title(img_bgr: np.ndarray, title: str) -> np.ndarray:
    dpi = 200.0
    h, w = img_bgr.shape[:2]
    fig_w = w / dpi
    fig_h = (h + 30) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.text(0.5, 0.03, title, ha="center", va="bottom", fontsize=10, transform=ax.transAxes)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(H, W, 4)
    plt.close(fig)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

def _concat_horz(panels: list[np.ndarray]) -> np.ndarray:
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h), interpolation=cv2.INTER_AREA)
              for p in panels]
    return cv2.hconcat(panels)

def _make_input_panel(u_hist, T, width=700, height=420):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.plot(np.arange(len(u_hist)), u_hist, lw=1.5)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("t")
    ax.set_ylabel("u(t) ~ U[0,1]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    bgr = _mpl_fig_to_bgr(fig)
    plt.close(fig)
    return bgr

def _wrap_title(combined_bgr: np.ndarray, title: str) -> np.ndarray:
    dpi = 200.0
    h, w = combined_bgr.shape[:2]
    fig_w = w / dpi
    fig_h = (h + 40) / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=14, transform=ax.transAxes)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(H, W, 4)
    plt.close(fig)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

# =================== Main: BIO vs STANDARD ESN ===================

def main(ri: int = RI, ci: int = CI, out_path: str = OUT_PATH, fps: int = FPS):
    # Load CE (BIO model)
    ce_W_bio, ce_ei = load_connectome(CE_ADJ_PATH, CE_EI_PATH)
    N = ce_W_bio.shape[0] if ce_W_bio is not None else N_DEFAULT
    if WS_K >= N:
        raise ValueError(f"WS_K must be < N. Got WS_K={WS_K}, N={N}")

    # BIO model selection
    rname_bio, rconf_bio = ROW_DEFS[ri]
    cname_bio, cconf_bio = COL_DEFS[ci]
    seed_bio = SEED + ri*37 + ci*101

    # BIO reservoir
    W_bio, Win_bio, _ = build_reservoir(
        feature_conn=rconf_bio["conn"],
        feature_weights=rconf_bio["weights"],
        feature_dale=cconf_bio["dale"],
        feature_sr=cconf_bio["sr"],
        N=N, ce_W_bio=ce_W_bio, ce_ei=ce_ei,
        ws_k=WS_K, sr_target=SR_TARGET_OPT,
        input_scale=INPUT_SCALE, seed=seed_bio,
    )

    # STANDARD ESN (right panel)
    # Use same N, same SR policy as BIO column.
    seed_esn = SEED + 9999 + ri*37 + ci*101
    W_esn, Win_esn = build_standard_esn(
        N=N,
        sparsity=ESN_SPARSITY,
        weight_std=ESN_WEIGHT_STD,
        feature_sr=cconf_bio["sr"],
        sr_target=SR_TARGET_OPT,
        input_scale=INPUT_SCALE,
        seed=seed_esn,
    )

    # Shared input sequence (exact MC probe)
    T_total = WASHOUT + T_TRAIN + T_TEST
    set_seed(SEED)
    u = torch.rand(T_total, 1, device=DEVICE)

    # States
    zb = torch.zeros(W_bio.shape[0], device=DEVICE)
    ze = torch.zeros(W_esn.shape[0], device=DEVICE)

    # Viewers
    names_bio = np.array([f"N{i}" for i in range(W_bio.shape[0])])
    names_esn = np.array([f"N{i}" for i in range(W_esn.shape[0])])

    class _DummyWorm:
        def __init__(self, W, names):
            self.W = W.astype(np.float32)
            self.N = W.shape[0]
            self.names = list(map(str, names))
            self.V = np.zeros(self.N, np.float32)
            self.V_vis = self.V.copy()
            self.spiked_vis = np.zeros(self.N, bool)
            self.muscle_mask = np.zeros(self.N, bool)
        @property
        def _edge_w(self): return self.W[self.W != 0.0]
        @property
        def _edge_ptr(self):
            ij = np.argwhere(self.W != 0.0)
            idx = np.arange(len(ij))
            return np.column_stack([idx, ij])

    viewer_bio = ConnectomeViewer(
        _DummyWorm(W_bio.detach().cpu().numpy(), names_bio),
        layout="kamada_groups", spread=1.0, pulse_size=2.5, group_gap=0.5,
        color_mode="heat", colormap="viridis", vmax=1.0, node_size=NODE_SIZE,
    )
    viewer_esn = ConnectomeViewer(
        _DummyWorm(W_esn.detach().cpu().numpy(), names_esn),
        layout="kamada_groups", spread=1.0, pulse_size=2.5, group_gap=0.5,
        color_mode="heat", colormap="viridis", vmax=1.0, node_size=NODE_SIZE,
    )

    os.makedirs("tmp_mc_frames", exist_ok=True)
    frames = []
    u_hist = []

    print(f"[bio]  {rname_bio} × {cname_bio}  seed={seed_bio}")
    print(f"[esn]  ER p={ESN_SPARSITY}, N(0,{ESN_WEIGHT_STD}^2), SR={cconf_bio['sr']}  seed={seed_esn}")

    for t in range(T_total):
        pre_b = (W_bio @ zb + (Win_bio @ u[t:t+1, :].T).squeeze())
        pre_e = (W_esn @ ze + (Win_esn @ u[t:t+1, :].T).squeeze())

        hb = torch.tanh(pre_b); zb = (1 - LEAK) * zb + LEAK * hb
        he = torch.tanh(pre_e); ze = (1 - LEAK) * ze + LEAK * he

        Vb = pre_b.detach().cpu().numpy().astype(np.float32)
        Ve = pre_e.detach().cpu().numpy().astype(np.float32)

        both_abs = np.abs(np.concatenate([Vb, Ve]))
        vmax = float(np.percentile(both_abs, VMAX_PCT))
        vmin = float(np.percentile(both_abs, VMIN_PCT))
        if not np.isfinite(vmax) or vmax < 1e-6: vmax = 1.0
        if not np.isfinite(vmin) or vmin >= vmax: vmin = 0.0
        if hasattr(viewer_bio, "norm"):
            viewer_bio.norm.vmin = vmin; viewer_bio.norm.vmax = vmax
        if hasattr(viewer_esn, "norm"):
            viewer_esn.norm.vmin = vmin; viewer_esn.norm.vmax = vmax

        viewer_bio.wc.V[:] = Vb; viewer_bio.wc.V_vis = Vb.copy(); viewer_bio.wc.spiked_vis[:] = False
        viewer_esn.wc.V[:] = Ve; viewer_esn.wc.V_vis = Ve.copy(); viewer_esn.wc.spiked_vis[:] = False

        viewer_bio.step(); viewer_bio.ax.axis("off")
        viewer_esn.step(); viewer_esn.ax.axis("off")

        panel_bio = _panel_title(_mpl_fig_to_bgr(viewer_bio.fig), f"BIO: {rname_bio} × {cname_bio}")
        panel_esn = _panel_title(_mpl_fig_to_bgr(viewer_esn.fig), "Std ESN: ER + Gaussian")

        u_hist.append(float(u[t].item()))
        left_panel = _make_input_panel(u_hist, T_total)

        combined = _concat_horz([left_panel, panel_bio, panel_esn])
        framed   = _wrap_title(combined, TITLE_TEXT)
        path = os.path.join("tmp_mc_frames", f"frame_{t:05d}.png")
        cv2.imwrite(path, framed)
        frames.append(path)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", audio=False)

    for p in frames:
        try: os.remove(p)
        except Exception: pass

    print(f"saved {out_path}  ({len(frames)} frames)")

if __name__ == "__main__":
    main(ri=RI, ci=CI, out_path=OUT_PATH, fps=FPS)

