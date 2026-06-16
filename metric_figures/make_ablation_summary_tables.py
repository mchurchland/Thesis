#!/usr/bin/env python3
"""Create slide-ready ablation summary tables from CV dispersion outputs."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

METRICS = ["MC", "IPC", "KR", "GR"]
SAME_THRESHOLD = 5.0

BG = "#f8fafc"
INK = "#0f172a"
MUTED = "#475569"
GRID = "#dbe3ec"
HEADER = "#33464d"
ROW_ALT = "#eef3f6"
ROW_BASELINE = "#e8f6f4"

LABELS = {
    "real": "C. elegans",
    "cel+randN": "CE Gaussian wt.",
    "er+randN": "ER Gaussian wt.",
    "ws_p0.1+randN": "WS Gaussian wt.",
    "local_sign": "Sign-pres. Gaussian",
    "local_sign+flat": "Sign-pres. uniform",
    "local_sign+binary": "Sign-pres. +/-1 wt.",
    "global_sign_pres": "+/-1 sign shuffle",
    "binary_base": "Unsigned binary wt.",
    "shuffle": "CE weight shuffle",
    "conn_shuf_only": "Connection shuffle only",
    "celW+connShuf": "Connection + weight shuffle",
    "binary_base_topology_shuffle": "Unsigned binary + topology shuffle",
    "binary+shuffle": "+/-1 conn. shuffle",
    "binary+conshuffle+wshuffle": "+/-1 conn. + sign shuffle",
}

INTERPRETATIONS = {
    "real": "Low-dispersion reference: biological topology, sign balance, and heterogeneous weights act together.",
    "cel+randN": "Preserving topology alone is not enough; dropping biological signs and magnitudes moves MC/KR/GR into a higher-CV regime.",
    "er+randN": "Random topology plus Gaussian weights loses the low-CV profile, showing that connectome structure still matters.",
    "ws_p0.1+randN": "Matches ER across metrics in the thesis; small-world organization alone does not recover invariance.",
    "local_sign": "With topology and signs fixed, random magnitudes behave like the uniform control; MC is closest to the connectome.",
    "local_sign+flat": "Uniform and Gaussian sign-preserving magnitudes are similar, but neither fully reproduces IPC/KR/GR.",
    "local_sign+binary": "Collapsing weights to +/-1 preserves much of IPC/KR/GR, but thesis equivalence tests say MC is not recovered.",
    "global_sign_pres": "Preserving only global E/I balance is close to local +/-1; exact edge-by-edge sign placement is less important for IPC/KR/GR.",
    "binary_base": "Low CV alone is not the full biological profile; this unsigned model was equivalent to the connectome only for IPC.",
    "shuffle": "Real-weight shuffles all move by nearly the same amount; disrupting heterogeneous weight placement lowers invariance.",
    "conn_shuf_only": "Matches the weight-shuffle effect, implying rewiring mostly acts by moving biological weights to new edges.",
    "celW+connShuf": "Key confirmation: adding weight shuffle after connection shuffle does not add a new effect; connection shuffle's effect is weight reassignment.",
    "binary_base_topology_shuffle": "In the unsigned binary control, IPC/KR/GR remain stable, while MC is the metric most sensitive to topology.",
    "binary+shuffle": "After weights are collapsed to +/-1, GR/KR/IPC are largely insensitive to rewiring; MC carries residual topology sensitivity.",
    "binary+conshuffle+wshuffle": "Adding sign shuffle after +/-1 rewiring changes little for GR/KR/IPC; MC remains the main topology-sensitive metric.",
}


def load_mean_cv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mean_cv = df.pivot_table(index="mode", columns="metric", values="dispersion", aggfunc="mean")
    return mean_cv.reindex(columns=METRICS)


def load_mean_cv_from_combined(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for (mode, src, shuffle_id), group in df.groupby(["mode", "src", "shuffle_id"]):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "mode": mode,
                    "src": src,
                    "shuffle_id": shuffle_id,
                    "metric": metric,
                    "dispersion": values.std() / (abs(values.mean()) + 1e-12),
                }
            )
    disp = pd.DataFrame(rows)
    mean_cv = disp.pivot_table(index="mode", columns="metric", values="dispersion", aggfunc="mean")
    return mean_cv.reindex(columns=METRICS)


def direction(value: float, baseline: float) -> tuple[str, float]:
    rel = ((value - baseline) / baseline) * 100.0
    rounded_rel = round(rel)
    if rounded_rel >= SAME_THRESHOLD:
        return "inc", rel
    if rounded_rel <= -SAME_THRESHOLD:
        return "dec", rel
    return "same", rel


def format_result(vals: pd.Series) -> str:
    return "MC {MC:.3f} | IPC {IPC:.3f}\nKR {KR:.3f} | GR {GR:.3f}".format(**vals.to_dict())


def format_change(vals: pd.Series, baseline: pd.Series) -> str:
    parts = []
    for metric in METRICS:
        label, rel = direction(float(vals[metric]), float(baseline[metric]))
        parts.append(f"{metric} {label} {rel:+.0f}%")
    return "; ".join(parts[:2]) + "\n" + "; ".join(parts[2:])


def make_row(
    med: pd.DataFrame,
    mode: str,
    *,
    baseline_mode: str,
    label: str | None = None,
    interpretation: str | None = None,
    baseline_text: str = "baseline",
    group_start: bool = False,
) -> dict[str, str]:
    vals = med.loc[mode]
    if mode == baseline_mode:
        change = baseline_text
    else:
        change = format_change(vals, med.loc[baseline_mode])
    return {
        "Ablation": label or LABELS.get(mode, mode),
        "Result on each metric": format_result(vals),
        "Change": change,
        "Interpretation": interpretation if interpretation is not None else INTERPRETATIONS.get(mode, ""),
        "_group_start": "1" if group_start else "",
    }


def build_rows(med: pd.DataFrame, modes: list[str], baseline_mode: str = "real") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for mode in modes:
        if mode not in med.index:
            continue
        rows.append(make_row(med, mode, baseline_mode=baseline_mode))
    return rows


def wrapped(text: str, width: int) -> str:
    lines = []
    for raw_line in str(text).splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=width, break_long_words=False))
    return "\n".join(lines)


def draw_table(
    rows: list[dict[str, str]],
    title: str,
    subtitle: str,
    out_stem: str,
    *,
    change_header: str = "Inc/dec/same\nvs C. elegans",
) -> None:
    fig, ax = plt.subplots(figsize=(13.4, 7.55), constrained_layout=False)
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.955, title, ha="left", va="top", fontsize=22, fontweight="bold", color=INK)
    ax.text(0.035, 0.905, subtitle, ha="left", va="top", fontsize=11.5, color=MUTED)

    x0, y_top = 0.035, 0.855
    table_w = 0.93
    header_h = 0.064
    row_h = 0.073 if len(rows) <= 9 else 0.064
    widths = [0.19, 0.235, 0.255, 0.32]
    headers = ["Ablation", "Result on each metric\n(mean CV)", change_header, "Interpretation"]
    wrap_widths = [22, 28, 35, 52]

    xs = [x0]
    for width in widths[:-1]:
        xs.append(xs[-1] + width * table_w)

    # Header.
    ax.add_patch(Rectangle((x0, y_top - header_h), table_w, header_h, facecolor=HEADER, edgecolor=HEADER, linewidth=1))
    for idx, header in enumerate(headers):
        ax.text(
            xs[idx] + 0.008,
            y_top - header_h / 2,
            header,
            ha="left",
            va="center",
            fontsize=10.7,
            fontweight="bold",
            color="white",
            linespacing=1.05,
        )

    # Rows.
    y = y_top - header_h
    for row_idx, row in enumerate(rows):
        y_next = y - row_h
        is_baseline = "baseline" in row["Change"].lower()
        fill = ROW_BASELINE if is_baseline else ("white" if row_idx % 2 else ROW_ALT)
        ax.add_patch(Rectangle((x0, y_next), table_w, row_h, facecolor=fill, edgecolor=GRID, linewidth=0.8))
        if row.get("_group_start") and row_idx > 0:
            ax.plot([x0, x0 + table_w], [y, y], color=HEADER, lw=2.2)

        values = [
            wrapped(row["Ablation"], wrap_widths[0]),
            wrapped(row["Result on each metric"], wrap_widths[1]),
            wrapped(row["Change"], wrap_widths[2]),
            wrapped(row["Interpretation"], wrap_widths[3]),
        ]
        for col_idx, value in enumerate(values):
            ax.text(
                xs[col_idx] + 0.008,
                y_next + row_h / 2,
                value,
                ha="left",
                va="center",
                fontsize=9.7 if col_idx == 3 else 10.1,
                color=INK,
                fontweight="bold" if col_idx == 0 else "normal",
                linespacing=1.05,
            )
        for col_idx in range(1, len(widths)):
            ax.plot([xs[col_idx], xs[col_idx]], [y_next, y], color=GRID, lw=0.8)
        y = y_next

    ax.text(
        0.035,
        0.035,
        f"Higher CV = less invariant; lower CV = more invariant. Inc/dec/same uses a +/-{SAME_THRESHOLD:.0f}% mean-CV threshold.",
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=MUTED,
    )

    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_csv(rows: list[dict[str, str]], out_stem: str) -> None:
    visible_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    pd.DataFrame(visible_rows).to_csv(OUT_DIR / f"{out_stem}.csv", index=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
        }
    )

    main_med = load_mean_cv(ROOT / "io_seperate/full_v2/dispersion_by_group.ALL.csv")
    main_modes = [
        "real",
        "cel+randN",
        "er+randN",
        "ws_p0.1+randN",
        "local_sign",
        "local_sign+flat",
        "local_sign+binary",
        "global_sign_pres",
        "binary_base",
    ]
    main_rows = build_rows(main_med, main_modes)
    draw_table(
        main_rows,
        "Ablation summary from invariance CV",
        "Mean coefficient of variation across the 48 hyperparameter settings; compared to the C. elegans baseline.",
        "ablation_cv_summary_table",
    )
    save_csv(main_rows, "ablation_cv_summary_table")

    real_shuf_med = load_mean_cv_from_combined(ROOT / "io_together/full_shuf_v1/combined.ALL.csv")
    shuf_rows = [
        make_row(
            real_shuf_med,
            "real",
            baseline_mode="real",
            label="C. elegans (real wt.)",
            baseline_text="real-wt baseline",
            group_start=True,
            interpretation="Real-weight reference for degree-preserving shuffle controls.",
        ),
        make_row(real_shuf_med, "shuffle", baseline_mode="real"),
        make_row(real_shuf_med, "conn_shuf_only", baseline_mode="real"),
        make_row(real_shuf_med, "celW+connShuf", baseline_mode="real"),
        make_row(
            real_shuf_med,
            "local_sign+binary",
            baseline_mode="local_sign+binary",
            label="+/-1 sign-pres. baseline",
            baseline_text="+/-1 baseline",
            group_start=True,
            interpretation="Magnitude heterogeneity is removed while preserving the original sign assignment.",
        ),
        make_row(
            real_shuf_med,
            "global_sign_pres",
            baseline_mode="local_sign+binary",
            label="+/-1 sign shuffle",
        ),
        make_row(
            real_shuf_med,
            "binary+shuffle",
            baseline_mode="local_sign+binary",
            label="+/-1 connection shuffle",
        ),
        make_row(
            real_shuf_med,
            "binary+conshuffle+wshuffle",
            baseline_mode="local_sign+binary",
            label="+/-1 conn. + sign shuffle",
        ),
        make_row(
            real_shuf_med,
            "binary_base",
            baseline_mode="binary_base",
            label="Unsigned binary baseline",
            baseline_text="binary baseline",
            group_start=True,
            interpretation="All nonzero edges have identical sign and magnitude, isolating topology most directly.",
        ),
        make_row(
            real_shuf_med,
            "binary_base_topology_shuffle",
            baseline_mode="binary_base",
            label="Unsigned binary topology shuffle",
        ),
    ]
    draw_table(
        shuf_rows,
        "Shuffle experiment summary from invariance CV",
        "Mean coefficient of variation from io_together/full_shuf_v1; each block is compared to its own experiment baseline.",
        "shuffle_cv_summary_table",
        change_header="Inc/dec/same\nvs block baseline",
    )
    save_csv(shuf_rows, "shuffle_cv_summary_table")

    for name in [
        "ablation_cv_summary_table.png",
        "ablation_cv_summary_table.pdf",
        "ablation_cv_summary_table.csv",
        "shuffle_cv_summary_table.png",
        "shuffle_cv_summary_table.pdf",
        "shuffle_cv_summary_table.csv",
    ]:
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
