# C. elegans reservoir computing experiments

Research code and results for comparing the *C. elegans* connectome with
synthetic and shuffled reservoir architectures. The experiments measure how
network topology, edge weights, and excitatory/inhibitory balance affect both
reservoir performance and robustness across hyperparameter choices.

The main metrics are:

- **MC** — linear memory capacity
- **IPC** — information processing capacity
- **KR** — kernel rank (state-space separation)
- **GR** — generalization rank (lower values indicate better generalization)

For each architecture, the pipeline sweeps spectral radius, leak rate, input
scale, and neuron bias. It then summarizes both mean metric performance and
invariance, primarily as the coefficient of variation (CV) over that grid.

> This is a research repository, not an installable Python package. The
> committed SLURM files encode the production experiment settings, while the
> committed result directories contain merged tables and generated figures
> from several analysis stages.

## Repository layout

| Path | Purpose |
|---|---|
| `inv_arc_test.py` | Main command-line entry point for architecture and sign sweeps |
| `reservoir_variants.py` | Reservoir variants, evaluation, and CSV output |
| `network_stats/` | Reservoir dynamics and MC, IPC, KR, GR implementations |
| `util/` | Connectome loading, graph construction, merging, and result-repair utilities |
| `Connectome/` | Processed 297-node adjacency, neuron names, and unknown-sign edge weights |
| `r_*.sbatch` | Production and methodology SLURM launchers |
| `Parameter_justification/` | MC/IPC delay, order, and methodology sensitivity analyses |
| `final_results/` | Curated merged results and thesis-ready figures |
| `final_results_GRKR_*`, `final_results_KR_*` | Rank-metric reruns and intermediate merged results |
| `metric_figures/` | Specialized plotting and table-generation scripts |
| `architecture_variant_figures/` | Network diagrams for the architecture variants |
| `logs/` | Committed SLURM logs from experiment runs |

## Setup

Python 3.10 or newer is recommended. The launchers expect a Conda environment
named `cel`, but any environment with the required packages will work.

```bash
conda create -n cel python=3.11
conda activate cel
```

Install a PyTorch build appropriate for the machine's CPU/CUDA setup, followed
by the remaining dependencies:

```bash
python -m pip install torch
python -m pip install numpy pandas scipy scikit-learn matplotlib networkx pingouin
```

Optional development and data-import dependencies are `pytest`, `xlrd`, and
`openpyxl`. Dependency versions are not currently locked, so record the
environment when producing new final results.

On systems where the default Matplotlib cache is not writable, set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-thesis
```

Run commands from the repository root so the local modules and relative data
paths resolve correctly.

## Quick start

This small CPU run evaluates the biological-weight connectome at one target
spectral radius. `--rank-only` computes KR and GR and leaves MC/IPC empty in a
new output file; omit it for all four metrics.

```bash
python inv_arc_test.py \
  --job real \
  --out-dir /tmp/cel-reservoir-smoke \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei "" \
  --unknown-sign-policy random_unknown_sign_matched \
  --ce-unknown-sign-weights Connectome/ce_unknown_sign_weights.npy \
  --rho-values 1.05 \
  --neuron-biases 0.0 \
  --n-repeats 1 \
  --rank-only \
  --cuda -1
```

The result is written to:

```text
/tmp/cel-reservoir-smoke/invariance_variants.csv
```

Use `python inv_arc_test.py --help` for every architecture, normalization,
partitioning, and sign-sweep option. An empty `--ce-ei` value is intentional
for the processed signed adjacency currently committed in `Connectome/`.

## Production experiments

The experiments are designed to run as SLURM arrays. Review the partition,
GPU, time limit, array size, result root, and connectome paths in a launcher
before submitting it.

```bash
sbatch r_data.sbatch
```

| Launcher | Experiment |
|---|---|
| `r_data.sbatch` | Main architecture comparison |
| `r_data_shuf.sbatch` | Topology and weight-shuffle controls |
| `r_data_shuf_baseline_rho.sbatch` | Shuffles normalized to the family baseline radius |
| `r_data_frac.sbatch` | Sign-fraction sweeps for connectome and matched ER networks |
| `r_sign_norm_ablation.sbatch` | Sign-balance normalization ablation |
| `r_triad_sign_fraction.sbatch` | Triad activity across sign fractions |
| `r_mc_delay.sbatch` | Memory-capacity delay sensitivity |
| `r_ipc_delay.sbatch` | IPC delay sensitivity |
| `r_ipc_order.sbatch` | IPC polynomial-order sensitivity |
| `r_ipc_methodology.sbatch` | Broader IPC methodology validation |

The primary launchers partition 200 seeded repeats over 20 array tasks. Each
task writes a provenance-tagged CSV below a `chunk_<id>/` directory. Fixed seed
offsets make repeat assignment, rank streams, and optional node subsets
reproducible across architectures.

Two normalization modes are available:

- `spectral_radius`: scale each generated reservoir by its own spectral radius.
- `original_radius`: scale controls using their architecture-family baseline,
  used by the fixed-baseline-radius shuffle experiment.

## Results and figures

Merged experiment directories generally contain:

```text
combined.ALL.csv              # all trial-level rows
mean_by_group.ALL.csv         # grouped metric means
dispersion_by_group.ALL.csv   # grouped invariance/CV summaries
```

`final_results/` is the curated result root. The similarly named
`final_results_GRKR_*` and `final_results_KR_*` directories preserve
intermediate and alternative rank calculations; do not assume they are the
canonical inputs for the thesis plots.

For the exact commands and input tables used to regenerate the final plots,
see [GRAPHING_README.md](GRAPHING_README.md). For example, the main
architecture density contours can be regenerated with:

```bash
python graph_hist.py \
  --combined final_results/main/combined.ALL.GRKR_erank.rank_updated.csv \
  --out-dir final_results/graphs/thesis/architecture \
  --show-cv-performance-contours
```

Architecture diagrams have their own instructions in
[architecture_variant_figures/README.md](architecture_variant_figures/README.md).

## Tests

Run the metric and rank-pipeline tests with:

```bash
python -m unittest network_stats.test_stats
```

The suite covers input-stream construction, KR/GR effective-rank behavior,
rank-only execution, and preservation of existing metrics during CSV updates.

## Reproducibility notes

- Keep the `src`, `sid`, `seed`, normalization, and hyperparameter columns when
  merging result files; downstream grouping and paired comparisons depend on
  them.
- Plot merged `combined.ALL.csv` files, not individual chunk outputs.
- Plotting scripts may create safely suffixed files such as `.v1` when an
  output already exists. Check the selected path before copying a figure into
  the thesis.
- Production launchers default to GPU execution. Pass `--cuda -1` for CPU
  execution when calling the Python entry point directly.
- The thesis LaTeX source is maintained separately; this repository contains
  the simulation, analysis, tabular results, and generated figure assets.
