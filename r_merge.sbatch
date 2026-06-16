#!/bin/bash
#SBATCH --job-name=merge_inv
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# Ensure we run from the directory where sbatch was submitted
cd "${SLURM_SUBMIT_DIR}"

# Avoid BLAS oversubscription; CPU-only job
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Paths (defaults match the merge script)
GLOB_SHUF="full_v1_experiment_full_chunks/chunk_*/invariance_variants.csv"
GLOB_VARIANTS="full_v1_experiment_full_chunks/chunk_*/invariance_variants.csv"
OUT_DIR="full_v1_experiment_full_merged"

# Run merge + plots (uses safe, non-destructive outputs)
#--glob-shuf "${GLOB_SHUF}" \
conda run -n cel python util/merge.py \
  --glob-variants "${GLOB_VARIANTS}" \
  --out-dir "${OUT_DIR}" \
  --bins 40 \
  --scatter-alpha 0.6
