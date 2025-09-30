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
    """
    W_bio, ei_labels = None, None
    if adj_path is not None and os.path.isfile(adj_path):
        W_bio = np.load(adj_path).astype(np.float32, copy=False)
        assert W_bio.ndim == 2 and W_bio.shape[0] == W_bio.shape[1], "CE adjacency must be square."
    if ei_path is not None and os.path.isfile(ei_path):
        ei_labels = np.load(ei_path).astype(np.float32)  # keep {-1,0,+1}
        if W_bio is not None:
            assert ei_labels.shape[0] == W_bio.shape[0], "EI labels length must match adjacency size."
    return W_bio, ei_labels


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
