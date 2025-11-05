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
GLOB_SHUF="experiment_full_chunks/chunk_*/bio_vs_shuffle_invariance.csv"
GLOB_VARIANTS="experiment_full_chunks/chunk_*/ce_real_reps.csv"
OUT_DIR="experiment_full_merged"

# Run merge + plots (uses safe, non-destructive outputs)
conda run -n cel python merge.py \
  --glob-shuf "${GLOB_SHUF}" \
  --glob-variants "${GLOB_VARIANTS}" \
  --out-dir "${OUT_DIR}" \
  --bins 40 \
  --scatter-alpha 0.6
