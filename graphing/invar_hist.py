import os
import pd
import numpy as np
import matplotlib as plt
from  util.graph_utils import _safe_path, _read_glob, _ensure_columns, _dispersion, _unique_hparam_rows

def plot_overlaid_arch_histograms(disp: pd.DataFrame, out_dir: str, bins: int):
    os.makedirs(out_dir, exist_ok=True)
    metrics = sorted(disp["metric"].unique())
    modes = sorted(disp["mode"].unique())

    # common bin edges per metric across all modes
    for m in metrics:
        all_vals = disp[disp["metric"] == m]["dispersion"].to_numpy()
        if all_vals.size == 0:
            continue
        lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        edges = np.linspace(lo, hi, bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        plt.figure(figsize=(10, 6))
        for mode in modes:
            s = disp[(disp["metric"] == m) & (disp["mode"] == mode)]["dispersion"].to_numpy()
            if s.size == 0:
                continue
            # counts normalized by number of entries → each curve integrates to 1
            counts, _ = np.histogram(s, bins=edges)
            frac = counts.astype(float) / max(len(s), 1)
            plt.plot(centers, frac, drawstyle="steps-mid", linewidth=1.8, alpha=0.95,
                     label=f"{mode} (N={len(s)})")

        plt.title(f"Invariance dispersion by architecture — {m}")
        plt.xlabel(f"dispersion = std({m}) / mean({m}) across hyper-params")
        plt.ylabel("fraction (normalized by N)")
        plt.legend(frameon=False, fontsize=9, ncol=2)
        plt.tight_layout()
        out_fig = _safe_path(os.path.join(out_dir, f"all_arch_hist_{m}.png"))
        plt.savefig(out_fig, dpi=200)
        plt.close()
        print(f"[saved] {out_fig}")


def plot_mc_vs_gr_all_arch(combined: pd.DataFrame, out_dir: str, alpha: float):
    if not {"MC","GR","mode"}.issubset(combined.columns):
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(11, 6))
    # light grey background points for all runs
    plt.scatter(combined["GR"], combined["MC"], s=8, c="#bbbbbb", alpha=alpha, label="all")
    # emphasize CE-real and CE-shuffle if present
    for cname, color in [("CE-real", "#1f77b4"), ("CE-shuffle", "#d62728")]:
        sub = combined[combined["mode"] == cname]
        if not sub.empty:
            sub_u = _unique_hparam_rows(sub)
            plt.scatter(sub_u["GR"], sub_u["MC"], s=36, alpha=0.95, label=cname, c=color)
    plt.title("MC vs GR across all architectures")
    plt.xlabel("GR (effective rank of Δstate)")
    plt.ylabel("MC (linear memory capacity)")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    out_fig = _safe_path(os.path.join(out_dir, "mc_vs_gr_all_arch.png"))
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"[saved] {out_fig}")