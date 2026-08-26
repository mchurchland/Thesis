# Graphing the final results

This is the canonical guide for regenerating figures from `final_results`.
Run every command from the repository root. The layouts referenced by
`~/Desktop/Reed/thesis_tex/thesis.tex` are the authoritative choices for the
thesis figures.

## Setup

```bash
cd ~/Thesis
conda activate cel
export MPLCONFIGDIR=/tmp/matplotlib-thesis-final
```

On the local workstation, replace `~/Thesis` with the path to this repository.
Setting `MPLCONFIGDIR` prevents Matplotlib cache-permission warnings on the
cluster.

The recommended destination for thesis figures is:

```text
final_results/graphs/thesis/
```

For the non-archival extended-abstract variants, run:

```bash
python make_extended_abstract_figures.py
```

This writes a separate 70%-contour, alternate-colour set to
`final_results/graphs/extended_abstract/` using only the rank-updated `_erank`
inputs listed below. It does not overwrite the thesis figures.

The plotting commands do not modify the merged result CSVs. `graph_hist.py`
may also create an `all_arch_hist_grid.png` and a CV table as side effects.

## Final merged inputs

| Experiment | Input |
|---|---|
| Main architecture comparison | `final_results/main/combined.ALL.GRKR_erank.rank_updated.csv` |
| Own-radius topology shuffles | `final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv` |
| Fixed-baseline-radius shuffles | `final_results/Rho_stuff/baseline_rho_shuf/combined.ALL.csv` |
| Sign sweep: C. elegans sign-matched | `final_results/sign_frac/cel_matched/combined.ALL.GRKR_erank.rank_updated.csv` |
| Sign sweep: predicted-polarity-only | `final_results/sign_frac/cel_removed/combined.ALL.GRKR_erank.rank_updated.csv` |
| Sign sweep: matched ER | `final_results/sign_frac/matched_er/combined.ALL.GRKR_erank.rank_updated.csv` |
| Sign-normalization versions | `final_results/sign_norm/{cel_matched,cel_removed,matched_er}/combined.ALL.csv` |
| Triad summary | `final_results/triad/triad_sign_fraction_group_summary.ALL.csv` |

Do not point plotting commands at `final_chunks`. Plot the merged
`combined.ALL.csv` files.

## 1. Main architecture islands (1 x 3)

This produces the thesis architecture figure with 50% CV/performance density
contours for IPC, KR, and MC. GR is intentionally reserved for the standalone
generalization-rank figure in Section 7 because lower GR is desirable.

```bash
python graph_hist.py \
  --combined final_results/main/combined.ALL.GRKR_erank.rank_updated.csv \
  --out-dir final_results/graphs/thesis/architecture \
  --show-cv-performance-contours

cp final_results/graphs/thesis/architecture/cv_performance_density_contours.png \
   final_results/graphs/thesis/architecture/50per.png
```

Primary output:

```text
final_results/graphs/thesis/architecture/50per.png
```

## 2. Topology-shuffle islands (3 x 3)

These are the row-specific triptychs. The rows use different baselines:

1. Real-valued C. elegans (`real`)
2. Sign-preserving signed-unit (`local_sign+binary`)
3. All-positive binary (`binary_base`)

### Own-radius normalization

```bash
python graph_hist.py \
  --combined final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv \
  --out-dir final_results/graphs/thesis/topology_shuffles \
  --show-cv-performance-contour-triptych

cp final_results/graphs/thesis/topology_shuffles/cv_performance_density_contours_triptych.png \
   final_results/graphs/thesis/topology_shuffles/shuf_all.png
```

### Fixed-baseline-radius normalization

```bash
python graph_hist.py \
  --combined final_results/Rho_stuff/baseline_rho_shuf/combined.ALL.csv \
  --out-dir final_results/graphs/thesis/baseline_rho_shuffles \
  --show-cv-performance-contour-triptych
```

Do **not** add `--triptych-axis-combined final_results/main/combined.ALL.csv`
for the final thesis versions. Letting each shuffle dataset determine its own
axes prevents the upper C. elegans MC contour from clipping.

Primary outputs:

```text
final_results/graphs/thesis/topology_shuffles/shuf_all.png
final_results/graphs/thesis/baseline_rho_shuffles/cv_performance_density_contours_triptych.png
```

## 3. Three family-specific CV-difference tables

`graph_hist.py` compares every mode present in its input with the requested
baseline. To make one table per triptych row, first create temporary CSVs that
contain only that row's architectures, then invoke `graph_hist.py` normally.

```bash
rm -rf /tmp/graph_hist_cv_tables
mkdir -p /tmp/graph_hist_cv_tables/{inputs,real,pm1,binary}

python - <<'PY'
import pandas as pd

source = pd.read_csv("final_results/shuf/combined.ALL.csv")
families = {
    "real": [
        "real", "shuffle", "celW+connShuf", "conn_shuf_only",
    ],
    "pm1": [
        "local_sign+binary", "binary+shuffle", "global_sign_pres",
        "binary+conshuffle+wshuffle",
    ],
    "binary": [
        "binary_base", "binary_base_topology_shuffle",
    ],
}

for name, modes in families.items():
    source[source["mode"].isin(modes)].to_csv(
        f"/tmp/graph_hist_cv_tables/inputs/{name}.csv",
        index=False,
    )
PY

python graph_hist.py \
  --combined /tmp/graph_hist_cv_tables/inputs/real.csv \
  --out-dir /tmp/graph_hist_cv_tables/real \
  --cv-baseline-mode real

python graph_hist.py \
  --combined /tmp/graph_hist_cv_tables/inputs/pm1.csv \
  --out-dir /tmp/graph_hist_cv_tables/pm1 \
  --cv-baseline-mode 'local_sign+binary'

python graph_hist.py \
  --combined /tmp/graph_hist_cv_tables/inputs/binary.csv \
  --out-dir /tmp/graph_hist_cv_tables/binary \
  --cv-baseline-mode binary_base

cp /tmp/graph_hist_cv_tables/real/cv_mean_differences_table.tex \
   final_results/graphs/thesis/topology_shuffles/cv_mean_differences_table_vs_real.tex

cp /tmp/graph_hist_cv_tables/pm1/cv_mean_differences_vs_local_sign_binary_table.tex \
   final_results/graphs/thesis/topology_shuffles/cv_mean_differences_vs_local_sign_binary_table.tex

cp /tmp/graph_hist_cv_tables/binary/cv_mean_differences_vs_binary_base_table.tex \
   final_results/graphs/thesis/topology_shuffles/cv_mean_differences_vs_binary_base_table.tex
```

The tables report control minus baseline. Positive `Delta CV` means the
control is less invariant. Confidence intervals are paired Student-t intervals
over matching trial groups.

## 4. Sign-fraction performance/CV curves

The thesis uses the existing `plot_frac_cv_meanline` function in
`graph_hist.py`. It is not currently exposed as a command-line flag, so invoke
the function directly. Each output stacks the 3D performance/CV curves above
the mean raw-spectral-radius curve for the same model's sign sweep. All lower
curves use the same color. The C. elegans panels mark the empirical sign
fraction; the ER panel omits that guide. None draws an original-network
spectral-radius reference:

```bash
python - <<'PY'
import graph_hist as graph

for dataset in ("cel_matched", "cel_removed", "matched_er"):
    input_csv = f"final_results/sign_frac/{dataset}/combined.ALL.GRKR_erank.rank_updated.csv"
    output_dir = f"final_results/graphs/thesis/sign_fraction/{dataset}"
    combined = graph._ensure_columns(graph._read_combined_csv(input_csv))
    dispersion = graph._compute_dispersion_table(combined, mode="cv")
    graph.plot_frac_cv_meanline(
        dispersion,
        combined,
        output_dir,
        bins=4,
        show=False,
        performance_scale="linear",
    )
PY
```

Outputs:

```text
final_results/graphs/thesis/sign_fraction/cel_matched/meanpoint_frac_cv_lines.png
final_results/graphs/thesis/sign_fraction/cel_removed/meanpoint_frac_cv_lines.png
final_results/graphs/thesis/sign_fraction/matched_er/meanpoint_frac_cv_lines.png
```

## 5. Sign-normalization plots (option 8)

First generate the summary CSVs used by the simplified plotting script:

The commands below calculate CV only at the nominal target
`rho_target=1.05`; mean performance continues to be averaged over the full
target grid.

To generate the alternative full-grid CV across spectral radius, leak, input
scale, and neuron bias, replace `--sign-norm-cv-rho-target 1.05` with
`--sign-norm-cv-include-rho` and use a separate output directory so both
summaries remain available.

```bash
python graph_hist.py \
  --combined final_results/sign_norm/cel_matched/combined.ALL.csv \
  --out-dir final_results/graphs/thesis/sign_normalization/data/cel_matched \
  --sign-norm-ablation \
  --sign-norm-prefix sign_test_og_cel \
  --sign-norm-cv-rho-target 1.05

python graph_hist.py \
  --combined final_results/sign_norm/cel_removed/combined.ALL.csv \
  --out-dir final_results/graphs/thesis/sign_normalization/data/cel_removed \
  --sign-norm-ablation \
  --sign-norm-prefix sign_test_og_cel \
  --sign-norm-cv-rho-target 1.05

python graph_hist.py \
  --combined final_results/sign_norm/matched_er/combined.ALL.csv \
  --out-dir final_results/graphs/thesis/sign_normalization/data/matched_er \
  --sign-norm-ablation \
  --sign-norm-prefix sign_test_er \
  --sign-norm-cv-rho-target 1.05
```

Then generate option 8 for all three datasets. `axis_limits=None` is
intentional: each version gets automatic, dataset-specific axes.

```bash
python - <<'PY'
from pathlib import Path
import metric_figures.make_sign_norm_simplified_plots as plots

root = Path.cwd()
plots.OUT_DIR = root / "final_results/graphs/thesis/sign_normalization"
plots.OUT_DIR.mkdir(parents=True, exist_ok=True)

base = plots.OUT_DIR / "data"
specs = [
    {
        "label": "C. elegans sign-matched",
        "path": base / "cel_matched",
        "marker_frac": 0.2425287356321839,
        "marker_label": r"$q_{\mathrm{orig}}$",
        "stem": "option_8_average_perf_cv",
    },
    {
        "label": "ER sign-matched",
        "path": base / "matched_er",
        "marker_frac": 0.2425287356321839,
        "marker_label": r"$q_{\mathrm{CE}}$",
        "stem": "option_8_average_perf_cv_er_matched",
    },
    {
        "label": "Removed connectome",
        "path": base / "cel_removed",
        "marker_frac": 0.2425287356321839,
        "marker_label": r"$q_{\mathrm{orig}}$",
        "stem": "option_8_average_perf_cv_removed",
    },
]

for info in specs:
    data = plots._load_dataset(info)
    plots.save_option_8_average_perf_cv(
        data,
        info=info,
        stem=info["stem"],
        axis_limits=None,
    )
PY

cp final_results/graphs/thesis/sign_normalization/option_8_average_perf_cv.png \
   final_results/graphs/thesis/sign_normalization/sign_norm.png
```

Both PNG and PDF versions are written by the simplified plotting script.

## 6. Raw-spectral-radius summary

Use the rank-updated `_erank` shuffle and sign-fraction CSVs. The summary parser
accepts both legacy mode names containing a normalization suffix and the current
files, which store `spectral_radius` in the `normalization` column:

```bash
python - <<'PY'
from pathlib import Path
import plot_raw_rho_performance_summary as plots

plots.SIGN_SWEEPS = [
    (
        "Matched C. elegans sweep",
        "cel_matched",
        "final_results/sign_frac/cel_matched/combined.ALL.GRKR_erank.rank_updated.csv",
        "sign_test_og_cel",
        "#d55e00",
    ),
    (
        "Removed C. elegans sweep",
        "cel_removed",
        "final_results/sign_frac/cel_removed/combined.ALL.GRKR_erank.rank_updated.csv",
        "sign_test_og_cel",
        "#e69f00",
    ),
    (
        "Matched ER sweep",
        "matched_er",
        "final_results/sign_frac/matched_er/combined.ALL.GRKR_erank.rank_updated.csv",
        "sign_test_er",
        "#56b4e9",
    ),
]

summary = plots.build_summary(
    "final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv",
    "final_results/sign_frac",
    max_sign_frac=0.5,
)

output = Path("final_results/graphs/thesis/raw_rho")
output.mkdir(parents=True, exist_ok=True)
summary.to_csv(output / "raw_rho_performance_summary.csv", index=False)
plots.plot_summary(
    summary,
    output,
    "raw_rho_performance_summary",
    y_scale="linear",
)
PY
```

## 7. Standalone generalization-rank synthesis

GR is excluded from figures whose vertical axis is labelled mean performance.
Generate its two-column architecture, sign-sweep, raw-radius, and shuffle
summary exclusively from the rank-updated `_erank` inputs. Sign-fraction
panel B uses the complete 0--100% E/I edge-balance sweep; the raw-radius
panels C and D retain sign-sweep conditions through 50%:

```bash
python plot_generalization_rank_summary.py
```

Outputs:

```text
final_results/graphs/thesis/generalization_rank/generalization_rank_summary.png
final_results/graphs/thesis/generalization_rank/generalization_rank_summary.pdf
final_results/graphs/thesis/generalization_rank/generalization_rank_summary.csv
```

In this figure, increasing GR means poorer generalization and increasing CV
means greater hyperparameter sensitivity.

## 8. Sign-sweep quick-look

The quick-look script has historical input paths hard-coded. The following
wrapper supplies the final merged data without changing the script:

```bash
python - <<'PY'
from pathlib import Path
import quick_sign_sweep_summary as quick
import plot_raw_rho_performance_summary as summary_plots

summary_plots.SIGN_SWEEPS = [
    (
        "Matched C. elegans sweep",
        "cel_matched",
        "",
        "sign_test_og_cel",
        "#d55e00",
    ),
    (
        "Removed C. elegans sweep",
        "cel_removed",
        "",
        "sign_test_og_cel",
        "#e69f00",
    ),
    (
        "Matched ER sweep",
        "matched_er",
        "",
        "sign_test_er",
        "#56b4e9",
    ),
]

summary = summary_plots.build_summary(
    "final_results/shuf/combined.ALL.csv",
    "final_results/sign_norm",
    max_sign_frac=1.0,
)

quick.OUT_DIR = Path("final_results/graphs/thesis/sign_sweep_quicklook")
quick.build_summary = lambda *_args, **_kwargs: summary.copy()
quick.main()
PY
```

For the separate C. elegans-only version, use the same wrapper but replace
`quick.main()` with:

```python
quick.main(celegans_only=True)
```

This writes `sign_sweep_quicklook_celegans_only.png` and
`sign_sweep_quicklook_celegans_only.pdf` without overwriting the combined
C. elegans/ER figure.

## 8. Triad sign-fraction plot

```bash
python graph_hist.py \
  --out-dir final_results/graphs/thesis/triad \
  --triad-sign-fraction-plot \
  --triad-summary-csv final_results/triad/triad_sign_fraction_group_summary.ALL.csv \
  --triad-plot-out final_results/graphs/thesis/triad/triad_sign_fraction.png \
  --triad-normalization all \
  --triad-scope all
```

## 9. Generic histogram grids

Running `graph_hist.py` with only `--combined` and `--out-dir` produces the
generic `all_arch_hist_grid.png`. For example:

```bash
python graph_hist.py \
  --combined final_results/main/combined.ALL.csv \
  --out-dir final_results/graphs/main

python graph_hist.py \
  --combined final_results/shuf/combined.ALL.csv \
  --out-dir final_results/graphs/topology_shuffles

python graph_hist.py \
  --combined final_results/baseline_rho_shuf/combined.ALL.csv \
  --out-dir final_results/graphs/baseline_rho_shuffles
```

The same command can be used on each `final_results/sign_frac/*/combined.ALL.csv`.
The sign-fraction inputs do not contain the main architecture baseline, so an
automatic CV-difference table may be skipped; the histogram itself is still
generated.

## Outputs not handled by graph_hist.py

The following result families use specialized schemas and cannot be passed to
`graph_hist.py`:

- `final_results/ipc_delay`
- `final_results/mc_delay`
- `final_results/ipc_order`

Their reports and plots are produced by:

- `Parameter_justification/ipc_delay_sensitivity.py`
- `Parameter_justification/mc_delay_sensitivity.py`
- `Parameter_justification/ipc_order_sweep_ce.py`

Running those scripts recomputes the corresponding experiments; they are not
plot-only readers for the existing CSVs. The existing IPC-order figure is
`final_results/ipc_order/ipc_by_order_contrib.png`.

## Common problems

### `No chunk CSVs found`

The merge path must match the directory that actually contains `chunk_*`.
For the final triad run, the chunks were under
`final_results/triad_sign_fraction_experiment`, not
`final_results/triad_sign_fraction`.

### A plot opens no window

The cluster uses a non-interactive Matplotlib backend. A warning that
`FigureCanvasAgg` cannot be shown is harmless if the command prints `[saved]`.

### A new file gets `.v1`, `.v2`, and so on

Several graphing functions deliberately avoid overwriting existing files.
Use the newly printed path, or remove/move the old generated figure before
rerunning if a stable filename is required.

### The 3 x 4 shuffle plot clips at the top

Do not use `--triptych-axis-combined` for the final shuffle figures. Automatic
limits from each shuffle dataset provide enough headroom.

### The wrong topology-shuffle plot was generated

The thesis topology plot is the 3 x 4 output from
`--show-cv-performance-contour-triptych`, not the 2 x 2 output from
`--show-cv-performance-contours`.

## Final checklist

Before using the figures in the thesis:

1. Confirm every input is under `final_results`, not an older `good_results`
   or `New_figs` directory.
2. Confirm the topology-shuffle plots are 3 x 4.
3. Confirm the topology table baselines are `real`, `local_sign+binary`, and
   `binary_base`.
4. Confirm each option-8 normalization figure uses automatic axes.
5. Visually inspect the upper-right MC contour for clipping.
6. Keep static network diagrams and the ESN schematic from `thesis_tex`; they
   are not generated from the merged experiment tables.
