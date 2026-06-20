#!/usr/bin/env python3
"""Make a minimal readout prediction figure from saved trace data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a simple true-vs-predicted signal trace.")
    p.add_argument(
        "--in-csv",
        default="ipc_order_sweep_ce_many/input_network_legendre_timeseries.csv",
        help="Input CSV produced by plot_input_network_legendre.py.",
    )
    p.add_argument(
        "--out-png",
        default="ipc_order_sweep_ce_many/readout_prediction_simple.png",
        help="Output PNG path.",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.genfromtxt(args.in_csv, delimiter=",", names=True)

    t = data["t"]
    true_signal = data["legendre_p3_odd"]
    prediction = data["readout_p3"]
    keep = np.isfinite(prediction)
    washout_end = float(t[keep][0])
    test_start = 52.0

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=args.dpi)
    ax.axvspan(t[0], washout_end, color="#4c78a8", alpha=0.08)
    ax.axvspan(washout_end, test_start, color="#59a14f", alpha=0.08)
    ax.axvspan(test_start, t[-1], color="#e15759", alpha=0.08)
    ax.plot(t, true_signal, color="#2f6f9f", linewidth=2.0, alpha=0.62, label="true past signal")
    ax.plot(t[keep], prediction[keep], color="#d84a3a", linewidth=2.0, alpha=0.62, label="readout prediction")
    ax.axvline(washout_end, ymin=0.22, ymax=0.98, color="#555555", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.axvline(test_start, color="#555555", linewidth=1.2, linestyle="--", alpha=0.5)
    ax.text((t[0] + washout_end) / 2, 0.94, "washout", ha="center", va="top", fontsize=11, transform=ax.get_xaxis_transform())
    ax.text((washout_end + test_start) / 2, 0.94, "training", ha="center", va="top", fontsize=11, transform=ax.get_xaxis_transform())
    ax.text((test_start + t[-1]) / 2, 0.94, "prediction", ha="center", va="top", fontsize=11, transform=ax.get_xaxis_transform())
    ax.set_xlabel("time")
    ax.set_ylabel("signal")
    ax.grid(alpha=0.2, linewidth=0.8)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[done] wrote {out_png}")


if __name__ == "__main__":
    main()
