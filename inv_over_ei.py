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
from mpl_toolkits.mplot3d import Axes3D

from reservoir_variants import evaluate_reservoir
from util.util import build_reservoir, load_connectome


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



def _build_col_params(
    sr_grid: Iterable[float],
    leak_grid: Iterable[float],
    u_grid: Iterable[float],
) -> list[tuple[float, float, float]]:

    return [(sr, leak, u) for sr in sr_grid for leak in leak_grid for u in u_grid]


def _split_indices(n_total: int, split: int, rank: int) -> list[int]:
    """
    Return the indices this rank should handle (array-job friendly).

    Supports both:
      - 0-based ranks in [0, split-1]
      - 1-based ranks in [1, split]
    """
    if n_total == 0:
        return []

    if split <= 1:
        return list(range(n_total))

    if not (0 <= rank < split):
        if 1 <= rank <= split:
            rank = rank - 1
        else:
            raise ValueError(
                f"--rank must be in [0, {split-1}] or [1, {split}] for --split={split}, got {rank}"
            )

    base = n_total // split
    rem = n_total % split
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


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
    ce_ei: np.ndarray | None,    ## get the positive weights of w
    col_params: list[tuple[float, float, float]],
    device: torch.device,
    ws_k: int,
    dispersion_mode: str,
    seed_base: int = 0,
    n_seeds: int = 1,
    seed_stride: int = 101,
    ei_balance: int = 0
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
    if ce_ei is None:
        raise ValueError("ce_ei is required for EI-balance sweeps.")
    all_ind = np.where(ce_ei != 0)[0]
    #neuron_test(ce_W_bio,ce_ei) 
    #ce_W_bio = gaus_m0_sign_pres(ce_W_bio,1.0, np.random.default_rng(0))
    for si in range(n_seeds):
            ei_seed = seed_base + si * seed_stride + 50_000
            rng_ei = np.random.default_rng(ei_seed)
            if ei_balance == 0:
                neg = np.array([], dtype=int)
            else:
                neg = rng_ei.choice(all_ind, replace=False, size=ei_balance)
            ce_ei_seed = np.abs(ce_ei).copy()
            ce_ei_seed[neg] = -1 * ce_ei_seed[neg]
            per_seed_vals = {k: [] for k in metrics}
            for ci, (target_sr, leak, in_scale) in enumerate(col_params):
                    cur_seed = seed_base + si * seed_stride + ci
                    rng_local = np.random.default_rng(cur_seed)
                    #ce_W_bio=cel_weight_global_sign_mixture_match(ce_W_bio,1.0,rng_local)
                    #W_mat = partial_weight_randomization(ce_W_bio, frac_replace, rng_local) partial_weight_randomization_stacked_gaus
                    #W_mat = gaus_m0_sign_pres(ce_W_bio, frac_replace, rng_local)

                    
                    W_mat = gaus_m0_ei_pres(ce_W_bio, frac_replace=1, rng = rng_local,ce_ei=ce_ei_seed)
                    #W_mat = degree_matched_shuffle_directed(ce_W_bio,frac_replace,rng_local)
                    try:
                        Wt, Win, _, _ = build_reservoir(
                            feature_conn="cel",
                            feature_weights="bio",
                            feature_dale="none",
                            target_sr=target_sr,
                            N=W_mat.shape[0],
                            ce_W_bio=W_mat,
                            ce_ei=ce_ei_seed,
                            ws_k=ws_k,
                            input_scale=in_scale,
                            seed=cur_seed,
                            drive_idx=None,
                            nnz_target=None,
                            DEVICE=device,
                        )
                        res = evaluate_reservoir(Wt, Win, leak, device)
                    except Exception:
                        res = dict(MC=np.nan, IPC=np.nan, KR=np.nan, GR=np.nan)
                    for k in metrics:
                        per_seed_vals[k].append(float(res[k]))
                    ##the 1 codes for frac replace
                    raw_rows.append((ei_balance, 1, si, target_sr, leak, in_scale, float(res["MC"]), float(res["IPC"]), float(res["KR"]), float(res["GR"])))
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
                "EI_bal",
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
        w.writerow(["EI_bal","frac_replace", "seed_id", "rho_target", "leak", "input_scale", "MC", "IPC", "KR", "GR"])
        w.writerows(rows)


def load_summary_rows(summary_csv: str) -> list[tuple]:
    import csv

    rows = []
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = [
            "EI_bal",
            "MC_mean",
            "MC_disp",
            "IPC_mean",
            "IPC_disp",
            "KR_mean",
            "KR_disp",
            "GR_mean",
            "GR_disp",
        ]
        if reader.fieldnames is None or any(k not in reader.fieldnames for k in required):
            raise ValueError(f"Summary CSV missing required columns: {required}")
        for row in reader:
            rows.append(
                (
                    int(float(row["EI_bal"])),
                    float(row["MC_mean"]),
                    float(row["MC_disp"]),
                    float(row["IPC_mean"]),
                    float(row["IPC_disp"]),
                    float(row["KR_mean"]),
                    float(row["KR_disp"]),
                    float(row["GR_mean"]),
                    float(row["GR_disp"]),
                )
            )
    rows.sort(key=lambda r: r[0])
    return rows


def load_summary_rows_from_glob(pattern: str) -> list[tuple]:
    import glob

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No summary CSVs matched: {pattern}")
    rows = []
    seen = {}
    for path in paths:
        for row in load_summary_rows(path):
            key = row[0]
            if key in seen:
                raise ValueError(f"Duplicate EI_bal {key} in {path} and {seen[key]}")
            seen[key] = path
            rows.append(row)
    rows.sort(key=lambda r: r[0])
    return rows


    

def make_ei_from_count(W_ce):

    W = W_ce.copy().astype(np.float32)
    W_ind_normal =np.sign(W) ## handel naan
    ei_label = W_ind_normal @ np.ones(W_ind_normal.shape[0])
    mixed_idx = np.where(ei_label == 0)[0]
    for idx in mixed_idx:
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

    return np.sign(ei_label).astype(np.float32), mixed_idx


def _tagged_name(name: str, tag: str | None) -> str:
    if not tag:
        return name
    root, ext = os.path.splitext(name)
    return f"{root}_{tag}{ext}"


def plot_dispersion(out_png: str, summary_rows: list[tuple]):
    ei_counts = [r[0] for r in summary_rows]
    mc_std = [r[2] for r in summary_rows]
    ipc_std = [r[4] for r in summary_rows]
    kr_std = [r[6] for r in summary_rows]
    gr_std = [r[8] for r in summary_rows]

    plt.figure(figsize=(7, 4), dpi=140)
    plt.plot(ei_counts, mc_std, marker="o", label="MC disp")
    plt.plot(ei_counts, ipc_std, marker="o", label="IPC disp")
    plt.plot(ei_counts, kr_std, marker="o", label="KR disp")
    plt.plot(ei_counts, gr_std, marker="o", label="GR disp")
    plt.xlabel("Inhibitory neuron count (EI balance)")
    plt.ylabel("Dispersion (per-seed over hyperparameter grid)")
    plt.title("Invariance loss vs EI balance (all edges redrawn)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")


def plot_metric_means(out_png: str, summary_rows: list[tuple]):
    ei_counts = [r[0] for r in summary_rows]
    mc_mean = [r[1] for r in summary_rows]
    ipc_mean = [r[3] for r in summary_rows]
    kr_mean = [r[5] for r in summary_rows]
    gr_mean = [r[7] for r in summary_rows]

    plt.figure(figsize=(7, 4), dpi=140)
    plt.plot(ei_counts, mc_mean, marker="o", label="MC mean")
    plt.plot(ei_counts, ipc_mean, marker="o", label="IPC mean")
    plt.plot(ei_counts, kr_mean, marker="o", label="KR mean")
    plt.plot(ei_counts, gr_mean, marker="o", label="GR mean")
    plt.xlabel("Inhibitory neuron count (EI balance)")
    plt.ylabel("Mean over seeds x hyperparameter grid")
    plt.title("Metric means vs EI balance (all edges redrawn)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")


def plot_mean_dispersion_3d(out_png: str, summary_rows: list[tuple]):
    ei_counts = [r[0] for r in summary_rows]
    mc_mean = [r[1] for r in summary_rows]
    mc_disp = [r[2] for r in summary_rows]
    ipc_mean = [r[3] for r in summary_rows]
    ipc_disp = [r[4] for r in summary_rows]
    kr_mean = [r[5] for r in summary_rows]
    kr_disp = [r[6] for r in summary_rows]
    gr_mean = [r[7] for r in summary_rows]
    gr_disp = [r[8] for r in summary_rows]

    fig = plt.figure(figsize=(7, 5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ei_counts, mc_disp, mc_mean, marker="o", label="MC")
    ax.plot(ei_counts, ipc_disp, ipc_mean, marker="o", label="IPC")
    ax.plot(ei_counts, kr_disp, kr_mean, marker="o", label="KR")
    ax.plot(ei_counts, gr_disp, gr_mean, marker="o", label="GR")
    ax.set_xlabel("Inhibitory neuron count (EI balance)")
    ax.set_ylabel("Dispersion (per-seed over hyperparameter grid)")
    ax.set_zlabel("Mean over seeds x hyperparameter grid")
    ax.set_title("Mean vs dispersion vs EI balance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved {out_png}")


def main():

    ap = argparse.ArgumentParser(description="Weight destruction ladder on CE adjacency.")
    ap.add_argument("--ce-adj", required=False, help="Path to C. elegans adjacency (npy).")
    ap.add_argument("--ce-ei", required=False, help="Path to C. elegans EI labels (npy).")
    ap.add_argument("--out-dir", default="ladder_results", help="Directory for CSVs/plots.")
    ap.add_argument("--ws-k", type=int, default=40, help="WS K (only used to match util.build_reservoir signature).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", type=int, default=None, help="CUDA device index; omit for auto.")
    ap.add_argument("--rho-targets", type=float, nargs="+", default=[ 0.8,1.0,1.2])
    ap.add_argument("--leaks", type=float, nargs="+", default=[0.2, 0.5,0.8])
    ap.add_argument("--input-scales", type=float, nargs="+", default=[0.5, 1.0,2.0])
    ap.add_argument("--n-seeds", type=int, default=1, help="Number of seeds to average per hyperparam point.")
    ap.add_argument("--seed-stride", type=int, default=101, help="Stride between seeds across columns.")
    ap.add_argument("--dispersion", choices=["cv", "std"], default="cv", help="Dispersion metric across hyperparams (per seed).")
    ap.add_argument("--split", type=int, default=1, help="Number of chunks for parallel runs.")
    ap.add_argument("--rank", type=int, default=0, help="This process's chunk index (0- or 1-based).")
    ap.add_argument("--out-tag", default="", help="Optional suffix for output filenames (useful for array jobs).")
    ap.add_argument("--plot-only", action="store_true", help="Read summary CSV and only draw the plot.")
    ap.add_argument("--summary-csv", default="", help="Summary CSV path for --plot-only (defaults to out-dir file).")
    ap.add_argument("--summary-glob", default="", help="Glob for summary CSVs (used with --plot-only).")
    ap.add_argument("--plot", action="store_true", help="Force plot generation even for partial chunks.")
    ap.add_argument("--no-plot", action="store_true", help="Disable plot generation.")
    args = ap.parse_args()

    out_tag = args.out_tag
    if not out_tag and args.split > 1:
        out_tag = f"chunk_{args.rank}"

    if args.plot_only:
        if args.summary_csv and args.summary_glob:
            ap.error("Use only one of --summary-csv or --summary-glob.")
        if args.summary_glob:
            summary_rows = load_summary_rows_from_glob(args.summary_glob)
        else:
            summary_csv = args.summary_csv
            if not summary_csv:
                summary_csv = os.path.join(args.out_dir, _tagged_name("ladder_summary_.csv", out_tag))
            summary_rows = load_summary_rows(summary_csv)
        os.makedirs(args.out_dir, exist_ok=True)
        out_png = os.path.join(args.out_dir, _tagged_name("gaus_ei_pres.png", out_tag))
        plot_dispersion(out_png, summary_rows)
        mean_png = os.path.join(args.out_dir, _tagged_name("gaus_ei_means.png", out_tag))
        plot_metric_means(mean_png, summary_rows)
        plot_mean_dispersion_3d(
            os.path.join(args.out_dir, _tagged_name("gaus_ei_3d.png", out_tag)),
            summary_rows,
        )
        return

    if not args.ce_adj or not args.ce_ei:
        ap.error("--ce-adj and --ce-ei are required unless --plot-only is set.")

    device = _pick_device(args.cuda)
    ce_W_bio, ce_ei, _ = load_connectome(args.ce_adj, args.ce_ei)
    ce_ei,mixed_idx = make_ei_from_count(ce_W_bio)      ## overrwrite because I dont like how the old one was handeled
    col_params = _build_col_params(args.rho_targets, args.leaks, args.input_scales)

    summary_rows = []
    raw_rows = []

    total_balances = np.count_nonzero(ce_ei != 0) + 1
    ei_balances = _split_indices(total_balances, args.split, args.rank)
    if not ei_balances:
        print("No EI balance indices assigned for this rank; exiting.")
        return

    for ei_bal in ei_balances:
        means, disp, rows = run_scores_for_fraction(
            ce_W_bio=ce_W_bio,

            ce_ei=ce_ei,
            col_params=col_params,
            device=device,
            ws_k=args.ws_k,
            dispersion_mode=args.dispersion,
            seed_base=args.seed + ei_bal * 10_000,
            n_seeds=args.n_seeds,
            seed_stride=args.seed_stride,
            ei_balance=ei_bal

        )
        summary_rows.append(
            (
                ei_bal,
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
    save_summary(os.path.join(args.out_dir, _tagged_name("ladder_summary_.csv", out_tag)), summary_rows)
    save_raw(os.path.join(args.out_dir, _tagged_name("ladder_raw_.csv", out_tag)), raw_rows)
    plot_full = len(ei_balances) == total_balances
    if not args.no_plot and (plot_full or args.plot):
        plot_dispersion(os.path.join(args.out_dir, _tagged_name("gaus_ei_pres.png", out_tag)), summary_rows)
        plot_metric_means(os.path.join(args.out_dir, _tagged_name("gaus_ei_means.png", out_tag)), summary_rows)
        plot_mean_dispersion_3d(
            os.path.join(args.out_dir, _tagged_name("gaus_ei_3d.png", out_tag)),
            summary_rows,
        )
    elif not plot_full:
        print("Skipping plot for partial chunk; pass --plot to force a partial plot.")

def degree_matched_shuffle_directed(A: np.ndarray, percent: float = 0.0,
                                    rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Degree-preserving double-edge swap randomization for directed graphs
    (Milo et al., 2002, Science 298:824-827).
    """
    if percent == 0.0:
        return A.copy().astype(np.float32)  # no change
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
    edges = np.argwhere(A)
    m = edges.shape[0]
    if m < 2:
        raise ValueError(f"Not enough edges to perform randomization. Found {m} edges. make sure the matrix is correct")
    if m* percent < 2:
        return A.copy().astype(np.float32)  # not enough edges to swap, return original
    #I dont need a for loop for this 
    max_retries = m*2 ##arbitrary
    retries = 0
    
    idx = rng.choice(m, size=np.floor(m * percent).astype(int), replace=False) ##i feel like this hsould be a stack
    #while not(idx.isempty) and retries < max_retries:
    i=0
    
    while i + 1 < len(idx) and retries < max_retries:
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

if __name__ == "__main__":
    main()
