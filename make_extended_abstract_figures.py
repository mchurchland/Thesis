#!/usr/bin/env python3
"""Regenerate the non-archival extended-abstract figure variants.

Every analysis input is a canonical rank-updated ``_erank`` CSV from
``final_results``.  The variants use a separate palette and 70% highest-density
contours, and are written outside the thesis/journal figure directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-extended-abstract")

import graph_hist as graph
import plot_generalization_rank_summary as gr_summary
import plot_raw_rho_performance_summary as rho_summary


MAIN_CSV = Path("final_results/main/combined.ALL.GRKR_erank.rank_updated.csv")
SHUFFLE_CSV = Path("final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv")
SIGN_INPUTS = {
    "cel_matched": Path(
        "final_results/sign_frac/cel_matched/combined.ALL.GRKR_erank.rank_updated.csv"
    ),
    "cel_removed": Path(
        "final_results/sign_frac/cel_removed/combined.ALL.GRKR_erank.rank_updated.csv"
    ),
    "matched_er": Path(
        "final_results/sign_frac/matched_er/combined.ALL.GRKR_erank.rank_updated.csv"
    ),
}
DEFAULT_OUT_DIR = Path("final_results/graphs/extended_abstract")
COLOR_SCHEME = "extended-abstract"
RAW_RHO_SIGN_SWEEPS = (
    (
        "Matched C. elegans sweep",
        "cel_matched",
        str(SIGN_INPUTS["cel_matched"]),
        "sign_test_og_cel",
        "#E07A5F",
    ),
    (
        "Removed C. elegans sweep",
        "cel_removed",
        str(SIGN_INPUTS["cel_removed"]),
        "sign_test_og_cel",
        "#4C78A8",
    ),
    (
        "Matched ER sweep",
        "matched_er",
        str(SIGN_INPUTS["matched_er"]),
        "sign_test_er",
        "#8F6BB3",
    ),
)


def _validate_inputs() -> None:
    for path in (MAIN_CSV, SHUFFLE_CSV, *SIGN_INPUTS.values()):
        if "_erank.rank_updated" not in path.name:
            raise ValueError(f"Expected a rank-updated _erank input, got: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)


def _joint_tables(path: Path):
    combined = graph._ensure_columns(graph._read_combined_csv(str(path)))
    dispersion = graph._compute_dispersion_table(combined, mode="cv")
    means = graph._compute_mean_table(combined)
    return combined, dispersion, means


def generate(out_dir: Path, contour_percent: float) -> list[Path]:
    _validate_inputs()
    contour_percent = float(contour_percent)
    if not (0.0 < contour_percent < 100.0):
        raise ValueError("contour_percent must be greater than 0 and less than 100.")

    outputs: list[Path] = []

    main, main_dispersion, main_means = _joint_tables(MAIN_CSV)
    architecture_dir = out_dir / "architecture"
    architecture_path = graph.plot_cv_performance_contours_2d(
        main_dispersion,
        main_means,
        str(architecture_dir),
        bins=40,
        show=False,
        contour_percent=contour_percent,
        baseline_mode="real",
        color_scheme=COLOR_SCHEME,
    )
    if architecture_path:
        outputs.append(Path(architecture_path))
    del main, main_dispersion, main_means

    shuffle, shuffle_dispersion, shuffle_means = _joint_tables(SHUFFLE_CSV)
    topology_dir = out_dir / "topology_shuffles"
    topology_path = graph.plot_cv_performance_contour_triptych(
        shuffle_dispersion,
        shuffle_means,
        shuffle,
        str(topology_dir),
        bins=40,
        show=False,
        contour_percent=contour_percent,
        color_scheme=COLOR_SCHEME,
    )
    if topology_path:
        outputs.append(Path(topology_path))
    del shuffle, shuffle_dispersion, shuffle_means

    for dataset, input_csv in SIGN_INPUTS.items():
        combined = graph._ensure_columns(graph._read_combined_csv(str(input_csv)))
        dispersion = graph._compute_dispersion_table(combined, mode="cv")
        sign_dir = out_dir / "sign_fraction" / dataset
        graph.plot_frac_cv_meanline(
            dispersion,
            combined,
            str(sign_dir),
            bins=4,
            show=False,
            performance_scale="linear",
            color_scheme=COLOR_SCHEME,
        )
        outputs.append(sign_dir / "meanpoint_frac_cv_lines.png")

    gr_outputs = gr_summary.create_figure(
        MAIN_CSV,
        SHUFFLE_CSV,
        gr_summary.EXTENDED_ABSTRACT_SIGN_SPECS,
        out_dir / "generalization_rank",
        "generalization_rank_summary",
        contour_percent=contour_percent,
        color_scheme=COLOR_SCHEME,
    )
    outputs.extend(gr_outputs)

    raw_rho = rho_summary.build_summary(
        SHUFFLE_CSV,
        Path("final_results/sign_frac"),
        max_sign_frac=0.5,
        sign_sweeps=RAW_RHO_SIGN_SWEEPS,
    )
    raw_rho_dir = out_dir / "raw_rho"
    raw_rho_dir.mkdir(parents=True, exist_ok=True)
    raw_rho_csv = raw_rho_dir / "raw_rho_performance_summary.csv"
    raw_rho.to_csv(raw_rho_csv, index=False)
    outputs.append(raw_rho_csv)
    outputs.extend(
        Path(path)
        for path in rho_summary.plot_summary(
            raw_rho,
            raw_rho_dir,
            "raw_rho_performance_summary",
            y_scale="linear",
            show=False,
            color_scheme=COLOR_SCHEME,
            sign_sweeps=RAW_RHO_SIGN_SWEEPS,
        )
    )

    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Expected figure outputs were not created: {missing}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--contour-percent",
        type=float,
        default=70.0,
        help="Highest-density percentage enclosed by contour figures (default: 70).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate(args.out_dir, args.contour_percent)
    print(
        f"[done] generated {len(outputs)} files in {args.out_dir} "
        f"using {args.contour_percent:g}% contours"
    )


if __name__ == "__main__":
    main()
