#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Filter combined.ALL.csv to CE-shuffle and celW+connShuf rows.")
    ap.add_argument("--input", default="experiment_full_merged/combined.ALL.csv",
                    help="Path to combined.ALL.csv")
    ap.add_argument("--output", default="experiment_full_merged/filtered_shuffles.csv",
                    help="Path to write filtered CSV (non-destructive if exists)")
    return ap.parse_args()

def _safe_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{root}.v{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Missing input: {args.input}")

    df = pd.read_csv(args.input)
    if "mode" not in df.columns:
        raise ValueError("Input CSV lacks required column: mode")

    keep = df["mode"].astype(str).isin(["CE-shuffle", "celW+connShuf"])
    out = df.loc[keep].copy()

    out_path = _safe_path(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[saved] {out_path}  (rows={len(out)})")

if __name__ == "__main__":
    main()
