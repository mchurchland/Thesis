from __future__ import annotations

"""
Weight-destruction ladder: gradually replace C. elegans weights with Gaussian draws
while keeping the CE adjacency fixed. Outputs per-fraction metrics and dispersion
(coefficient of variation across the hyperparameter grid) and a plot of dispersion vs frac_replace.
"""
import warnings
import argparse
import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reservoir_variants import evaluate_reservoir
from util.util import build_reservoir, load_connectome,degree_matched_shuffle_directed 


def partial_weight_randomization(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)
    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    k = int(frac_replace * len(idx))

    # match cel+randN: N(0, 1) on existing edges
    new_vals = rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32)

    sel = (nz[0][idx[:k]], nz[1][idx[:k]])
    W[sel] = new_vals
    return W
def partial_weight_randomization_stacked_gaus(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)
    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    k = int(frac_replace * len(idx))

    # match cel+randN: N(0, 1) on existing edges
    signs = rng.choice(np.array([-1,1],dtype=np.float32),size=k)
    new_vals = np.abs(rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32)) 

    sel = (nz[0][idx[:k]], nz[1][idx[:k]])
    W[sel] = new_vals * signs
    return W
def gaus_m0_sign_pres(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)

    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    k = int(frac_replace * len(idx))

    sel = (nz[0][idx[:k]], nz[1][idx[:k]])
    v = W[sel] ## v is a "pointer" to the weights that are selected
    sel_p = v > 0 ## get the positive weights of the selection
    sel_n = v < 0 ## get the negative weights of the selection


    # match cel+randN: N(0, 1) on existing edges
    num_pos = int(sel_p.sum())
    num_neg = int(sel_n.sum())
    if num_pos:
        new_vals_p = np.abs(rng.normal(loc=0.0, scale=1.0, size=num_pos).astype(np.float32))
        v[sel_p] = new_vals_p
    if num_neg:
        new_vals_n = -np.abs(rng.normal(loc=0.0, scale=1.0, size=num_neg).astype(np.float32))
        v[sel_n] = new_vals_n
    return W

def gaus_m0_ei_pres(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator,
                                 ce_ei: np.ndarray) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)
    W_bool = W_ce.copy().astype(np.bool_).astype(np.float32)
    diag = np.diag(ce_ei) ## put the +1/-1 on the diag of a 299x299 matrix
    ei_signs = np.matmul(diag,W_bool) ##im fucking cool ash
    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    k = int(frac_replace * len(idx))
    sel = (nz[0][idx[:k]], nz[1][idx[:k]]) ## these are rows and columns


    new_vals = np.abs(rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32))
    W[sel] = new_vals * ei_signs[sel]
    return W

def gaus_m0_ei_pres_mixed_rand(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator,
                                 ce_ei: np.ndarray,
                                 mixed_idx: np.ndarray) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)
    W_bool = W_ce.copy().astype(np.bool_).astype(np.float32)
    diag = np.diag(ce_ei) ## put the +1/-1 on the diag of a 299x299 matrix
    ei_signs = np.matmul(diag,W_bool) ##im fucking cool ash
    ei_signs[mixed_idx] = rng.choice([-1, 1], size=(mixed_idx.shape[0],W.shape[1]))
    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    k = int(frac_replace * len(idx))
    sel = (nz[0][idx[:k]], nz[1][idx[:k]]) ## these are rows and columns


    new_vals = np.abs(rng.normal(loc=0.0, scale=1.0, size=k).astype(np.float32))
    W[sel] = new_vals * ei_signs[sel]
    return W

def cel_weight_sample(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)
    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    k = int(frac_replace * len(idx))
    
    # match cel+randN: N(0, 1) on existing edges
    new_vals = rng.normal(loc = W[nz].mean(), scale=W[nz].std(ddof=0), size=k).astype(np.float32)

    sel = (nz[0][idx[:k]], nz[1][idx[:k]])
    W[sel] = new_vals
    return W

def cel_weight_local_sign_mixture_match(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)

    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)
    k = int(frac_replace * len(idx))

    sel = (nz[0][idx[:k]], nz[1][idx[:k]])
    v = W[sel] ## v is a "pointer" to the weights that are selected
     
    sel_p = v > 0 ## get the positive weights of the selection
    w_p = W[W > 0] ## get the positive weights of w

    sel_n = v < 0 ## get the negative weights of the selection
    w_n = W[W < 0] ## get the negative weights of w
    
    # match cel+randN: N(0, 1) on existing edges
    num_pos = int(sel_p.sum())
    num_neg = int(sel_n.sum())
    if num_pos:
        new_vals_p = rng.normal(loc = w_p.mean(), scale = w_p.std(ddof=0), size= num_pos).astype(np.float32)
        v[sel_p] = new_vals_p
    if num_neg:
        new_vals_n = rng.normal(loc = w_n.mean(), scale = w_n.std(ddof=0), size= num_neg).astype(np.float32)
        v[sel_n] = new_vals_n
    return W

def cel_weight_global_sign_mixture_match(W_ce: np.ndarray,
                                 frac_replace: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """
    Replace a fraction of existing C. elegans synaptic weights with Gaussian draws on the fixed topology
    """
    W = W_ce.copy().astype(np.float32)
    
    nz = np.nonzero(W)
    idx = np.arange(len(nz[0]))
    rng.shuffle(idx)


    num_pos = int((W[nz] > 0).sum())
    num_neg = int((W[nz] < 0).sum())
    frac = num_pos/(num_pos+num_neg) 
    k = int(frac_replace * len(idx))
    k_pos = int(round(k * frac))
    k_neg = k - k_pos
    assert k_neg+k_pos==k

    sel_p = (nz[0][idx[:k_pos]], nz[1][idx[:k_pos]])
    sel_n = (nz[0][idx[k_pos:k_pos+k_neg]], nz[1][idx[k_pos:k_pos+k_neg]])
     
    w_p = W[W > 0] ## get the positive weights of w
    w_n = W[W < 0] ## get the negative weights of w
    # match cel+randN: N(0, 1) on existing edges

    if num_pos:
        new_vals_p = rng.normal(loc = w_p.mean(), scale = w_p.std(ddof=0), size= k_pos).astype(np.float32)
        W[sel_p] = new_vals_p
    if num_neg:
        new_vals_n = rng.normal(loc = w_n.mean(), scale = w_n.std(ddof=0), size= k_neg).astype(np.float32)
        W[sel_n] = new_vals_n
    return W



def _build_col_params(
    sr_grid: Iterable[float],
    leak_grid: Iterable[float],
    u_grid: Iterable[float],
) -> list[tuple[float, float, float]]:

    return [(sr, leak, u) for sr in sr_grid for leak in leak_grid for u in u_grid]


def _pick_device(cuda_index: int | None) -> torch.device:
    if cuda_index is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cuda_index >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_index}")
    return torch.device("cpu")


def _dispersion(arr: np.ndarray, mode: str) -> float:
    """
    Compute dispersion (std or CV) with safe mean handling for CV (Sokal & Rohlf, 1995, Biometry 3rd ed.).
    """
    arr = np.asarray(arr, float)
    if mode == "std":
        return float(np.nanstd(arr))
    m = float(np.nanmean(arr))
    s = float(np.nanstd(arr))
    return s / (abs(m) + 1e-12)


def run_scores_for_fraction(
    ce_W_bio: np.ndarray,
    frac_replace: float,
    ce_ei: np.ndarray | None,    ## get the positive weights of w
    col_params: list[tuple[float, float, float]],
    device: torch.device,
    ws_k: int,
    dispersion_mode: str,
    seed_base: int = 0,
    n_seeds: int = 1,
    seed_stride: int = 101,
) -> tuple[dict[str, float], dict[str, float], list[tuple]]:
    """
    Sweep seeds/hyperparameters to compute MC/IPC/KR/GR reservoir metrics, following standard RC
    evaluations (Jaeger, 2002, GMD Report 152; Dambre et al., 2012, Sci. Rep. 2:514; Legenstein &
    Maass, 2007, Neural Networks 20:323-334).
    Return (means across all seed x hparam runs, dispersion averaged over seeds, raw rows). Dispersion
    is computed per-seed across the hyperparameter grid, then averaged over seeds.
    """
    metrics = ("MC", "IPC", "KR", "GR")
    # per-seed lists of per-hparam values
    seed_vals: dict[str, list[list[float]]] = {k: [] for k in metrics}
    raw_rows: list[tuple] = []
    #plot_cel_dist(ce_W_bio,np.random.default_rng(0))
    ce_ei,mixed_idx = make_ei_from_count(ce_W_bio) ## overrwrite because I dont like how the old one was handeled

    #neuron_test(ce_W_bio,ce_ei)
    #ce_W_bio = gaus_m0_sign_pres(ce_W_bio,1.0, np.random.default_rng(0))
    for si in range(n_seeds):
        per_seed_vals = {k: [] for k in metrics}
        for ci, (target_sr, leak, in_scale) in enumerate(col_params):
            cur_seed = seed_base + si * seed_stride + ci
            rng_local = np.random.default_rng(cur_seed)
            #ce_W_bio=cel_weight_global_sign_mixture_match(ce_W_bio,1.0,rng_local)
            #W_mat = partial_weight_randomization(ce_W_bio, frac_replace, rng_local) partial_weight_randomization_stacked_gaus
            #W_mat = gaus_m0_sign_pres(ce_W_bio, frac_replace, rng_local)
            
            #W_mat = gaus_m0_ei_pres(ce_W_bio, frac_replace, rng_local,ce_ei)
            #W_mat = gaus_m0_ei_pres_mixed_rand(ce_W_bio, frac_replace, rng_local,ce_ei,mixed_idx) # pyright: ignore[reportArgumentType]
            #W_mat = degree_matched_shuffle_directed(ce_W_bio,frac_replace,rng_local)
            Wt, Win, _, _ = build_reservoir(
                feature_conn="local_sign",
                    feature_weights="local_sign",
                    target_sr=target_sr,
                    N=None,
                    ce_W_bio=ce_W_bio,
                    ce_ei=None,
                    ws_k=ws_k,
                    input_scale=in_scale,
                    seed=cur_seed,
                    drive_idx=None,
                    nnz_target=None,
                    DEVICE=device,
                )
            res = evaluate_reservoir(Wt, Win, leak, device)
            for k in metrics:
                per_seed_vals[k].append(float(res[k]))
            raw_rows.append((frac_replace, si, target_sr, leak, in_scale, float(res["MC"]), float(res["IPC"]), float(res["KR"]), float(res["GR"])))
        for k in metrics:
            seed_vals[k].append(per_seed_vals[k])

    # means over all seed x hparam samples
    means = {}
    for k in metrics:
        flat = [v for seed_list in seed_vals[k] for v in seed_list]
        means[k] = float(np.nanmean(flat))

    # dispersion per seed, then averaged
    disp = {}
    for k in metrics:
        per_seed_disp = [_dispersion(np.asarray(seed_list, float), dispersion_mode) for seed_list in seed_vals[k]]
        disp[k] = float(np.nanmean(per_seed_disp))

    return means, disp, raw_rows


def save_summary(
    out_csv: str,
    rows: list[tuple],
):
    import csv

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frac_replace",
                "MC_mean",
                "MC_disp",
                "IPC_mean",
                "IPC_disp",
                "KR_mean",
                "KR_disp",
                "GR_mean",
                "GR_disp",
            ]
        )
        w.writerows(rows)


def save_raw(out_csv: str, rows: list[tuple]):
    """Write raw seed x hyperparameter rows to CSV (Shafranovich, 2005, IETF RFC 4180)."""
    import csv

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frac_replace", "seed_id", "rho_target", "leak", "input_scale", "MC", "IPC", "KR", "GR"])
        w.writerows(rows)

def plot_cel_dist(cel,rng,bins=50,samples = 1000):
    nz = np.nonzero(cel)
    w = cel[nz]
    #cel_normal = rng.normal(loc = cel[nz].mean(), scale =  cel[nz].std(ddof=0), size= 1000).astype(np.float32)
    w_p =  cel[cel > 0]
    w_n =  cel[cel > 0]
    cel_normal_p = rng.normal(loc = w_p.mean(), scale = w_p.std(ddof=0), size= samples).astype(np.float32)
    cel_normal_n = rng.normal(loc = w_n.mean(), scale = w_n.std(ddof=0), size= samples).astype(np.float32)
    fig, ax = plt.subplots()
    lo = min(w.min(), cel_normal_n.min())
    hi = max(w.max(), cel_normal_p.max())
    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots()
    ax.hist(cel_normal_n, bins=edges, density=True, alpha=0.3, color="blue", label="Normal fit neg")
    ax.hist(cel_normal_p, bins=edges, density=True, alpha=0.3, color="red", label="Normal fit pos")

    ax.hist(w,          bins=edges, density=True, alpha=0.3, color="green",  label="C. elegans (nonzero)")
    ax.legend()
    #ax.set_yscale('log')
    fig.savefig("cel_compar.png") 
    plt.close(fig)
    quit()

def make_ei_from_sum(W_ce):
    W = W_ce.copy().astype(np.float32)
    row_sum = W @ np.ones(W.shape[0])
    for idx in np.where(row_sum == 0)[0]:
        if not np.any(W[idx]):
            continue
        else:
            pos_count = (W[idx]>0).sum()
            neg_count = (W[idx]<0).sum()
            if pos_count == neg_count:
                row_sum[idx] = 0 ##gross but lets just assume if both sum is the same and pos_neg count is same its non
            else:
                row_sum[idx] = pos_count-neg_count
    return np.sign(row_sum) 
    
    

def make_ei_from_count(W_ce):

    W = W_ce.copy().astype(np.float32)
    W_ind_normal =np.sign(W) ## handel naan
    ei_label = W_ind_normal @ np.ones(W_ind_normal.shape[0])
    ei_mixed = np.where(ei_label == 0)[0]
    for idx in ei_mixed:
        if not np.any(W[idx]):
            continue
        else:
            r_sum = W[idx].sum()
            if r_sum==0:
                warnings.warn("When generating EI from count you had a row that had an equal number of\
                               positive and negative weights whos sum was 0,\
                               we set this EI value to 0 but this could change topology of the network\
                               you may want to rewrite this code if this behavior is not suitable for you")
                #this dosent happen in the celegan connectome and is just here defensivly
                #row_sum[idx] = 0 ##gross but lets just assume if both sum is the same and pos_neg count is same its non
                continue ## does the same thing as the line above
            else:
                ei_label[idx] = r_sum

    return np.sign(ei_label).astype(np.float32), ei_mixed


def neuron_test(W_ce,ce_ei):    
    print(ce_ei.shape)
    W = W_ce.copy().astype(np.float32)
    c_pos,c_neg,c_zero,c_both = 0,0,0,0
    print(len(W))
    
    print(len(np.where(ce_ei==0))) ###need to do a pause here and verify that everything lines up
    #quit()
    for e,nur in enumerate(W):
        pos_conn_count = (nur>0).sum()
        neg_conn_count = (nur<0).sum()


        if pos_conn_count>0 and neg_conn_count>0:
            c_both+=1

        elif neg_conn_count > 0:
            c_neg+=1
        elif pos_conn_count > 0:
            c_pos+=1

        else:
            c_zero+=1
            pass ## neurons with no outgoing connections
            print(nur,nur.sum()==0)
    print(
        f"pos={c_pos:6d} | "
        f"neg={c_neg:6d} | "
        f"both={c_both:6d} | "
        f"zero={c_zero:6d} | "
        f"total(nonzero)={c_neg + c_pos + c_both:6d} | "
        f"total={c_neg + c_pos + c_both + c_zero:6d}"
    )
    quit()


def plot_dispersion(out_png: str, summary_rows: list[tuple]):
    fracs = [r[0] for r in summary_rows]
    mc_std = [r[2] for r in summary_rows]
    ipc_std = [r[4] for r in summary_rows]
    kr_std = [r[6] for r in summary_rows]
    gr_std = [r[8] for r in summary_rows]

    plt.figure(figsize=(7, 4), dpi=140)
    plt.plot(fracs, mc_std, marker="o", label="MC disp")
    plt.plot(fracs, ipc_std, marker="o", label="IPC disp")
    plt.plot(fracs, kr_std, marker="o", label="KR disp")
    plt.plot(fracs, gr_std, marker="o", label="GR disp")
    plt.xlabel("Fraction of CE connections redrawn from gaus distribution")
    plt.ylabel("Dispersion (per-seed over hyperparameter grid)")
    plt.title("Invariance loss as CE weights are randomized, preserve Original EI balance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")


def main():

    ap = argparse.ArgumentParser(description="Weight destruction ladder on CE adjacency.")
    ap.add_argument("--ce-adj", required=True, help="Path to C. elegans adjacency (npy).")
    ap.add_argument("--ce-ei", required=True, help="Path to C. elegans EI labels (npy).")
    ap.add_argument("--out-dir", default="ladder_results", help="Directory for CSVs/plots.")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--ws-k", type=int, default=40, help="WS K (only used to match util.build_reservoir signature).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", type=int, default=None, help="CUDA device index; omit for auto.")
    ap.add_argument("--rho-targets", type=float, nargs="+", default=[0.6, 0.8, 0.95, 1.05])
    ap.add_argument("--leaks", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    ap.add_argument("--input-scales", type=float, nargs="+", default=[0.1, 0.5, 1.0, 1.5])
    ap.add_argument("--n-seeds", type=int, default=1, help="Number of seeds to average per hyperparam point.")
    ap.add_argument("--seed-stride", type=int, default=101, help="Stride between seeds across columns.")
    ap.add_argument("--dispersion", choices=["cv", "std"], default="cv", help="Dispersion metric across hyperparams (per seed).")
    args = ap.parse_args()

    device = _pick_device(args.cuda)
    ce_W_bio, ce_ei, _ = load_connectome(args.ce_adj, args.ce_ei)
    neuron_test(ce_ei=ce_ei,W_ce=ce_W_bio)
    quit()
    col_params = _build_col_params(args.rho_targets, args.leaks, args.input_scales)

    summary_rows = []
    raw_rows = []

    for fi, frac in enumerate(args.fractions):
        means, disp, rows = run_scores_for_fraction(
            ce_W_bio=ce_W_bio,
            frac_replace=frac,
            ce_ei=ce_ei,
            col_params=col_params,
            device=device,
            ws_k=args.ws_k,
            dispersion_mode=args.dispersion,
            seed_base=args.seed + fi * 10_000,
            n_seeds=args.n_seeds,
            seed_stride=args.seed_stride,
        )
        summary_rows.append(
            (
                frac,
                means["MC"],
                disp["MC"],
                means["IPC"],
                disp["IPC"],
                means["KR"],
                disp["KR"],
                means["GR"],
                disp["GR"],

            )
        )

        raw_rows.extend(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    save_summary(os.path.join(args.out_dir, "ladder_summary_2.csv"), summary_rows)
    save_raw(os.path.join(args.out_dir, "ladder_raw_2.csv"), raw_rows)
    plot_dispersion(os.path.join(args.out_dir, "gaus_ei_pres_mixed_rand.png"), summary_rows)



if __name__ == "__main__":
    main()
