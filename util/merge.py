#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge ALL outputs (CE real/shuffle + variants) into ONE CSV and plot ALL model
architectures on the SAME plots in the SAME style (histograms of dispersion).

Non-destructive: never deletes/modifies sources; uses safe suffixed outputs.

Inputs (defaults)
  experiment_full_chunks/chunk_*/bio_vs_shuffle_invariance.csv    # CE real vs shuffle
  experiment_full_chunks/chunk_*/invariance_variants.csv          # other architectures

Outputs (defaults)
  experiment_full_merged/combined.ALL.csv                         # row-wise merged runs
  experiment_full_merged/dispersion_by_group.ALL.csv              # per-group dispersion table
  experiment_full_merged/all_arch_hist_<MC|IPC|KR|GR>.png         # overlaid histograms per metric
  experiment_full_merged/mc_vs_gr_all_arch.png                    # scatter across all modes
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util.graph_utils import _safe_path, _read_glob, _build_combined, _compute_dispersion_table

# ---------------------------- CLI ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Merge CE real/shuffle + variants and plot all together.")
    ap.add_argument("--glob-shuf",
                    default="experiment_full_chunks/chunk_*/bio_vs_shuffle_invariance.csv",
                    help="Glob for CEL+bioW vs shuffled CSVs.")
    ap.add_argument("--glob-variants",
                    default="experiment_full_chunks/chunk_*/ce_real_reps.csv",
                    help="Glob for per-chunk variants CSVs.")
    ap.add_argument("--out-dir",
                    default="experiment_full_merged",
                    help="Directory to write merged CSVs and figures.")
    ap.add_argument("--bins", type=int, default=40,
                    help="Histogram bins for dispersion plots.")
    ap.add_argument("--scatter-alpha", type=float, default=0.55,
                    help="Alpha for MC-vs-GR scatter.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_shuf = _read_glob(args.glob_shuf)
    df_var = _read_glob(args.glob_variants)

    combined = _build_combined(df_shuf, df_var)
    out_comb = _safe_path(os.path.join(args.out_dir, "combined.ALL.csv"))
    combined.to_csv(out_comb, index=False)
    print(f"[saved] {out_comb}  (rows={len(combined)})")

    disp = _compute_dispersion_table(combined)
    out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    disp.to_csv(out_disp, index=False)
    print(f"[saved] {out_disp}  (rows={len(disp)})")

if __name__ == "__main__":
    main()
