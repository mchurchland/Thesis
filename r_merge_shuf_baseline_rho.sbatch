#!/bin/bash
#SBATCH --job-name=merge_base_rho
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUTROOT="${OUTROOT:-baseline_rho_topology_shuffles}"
MERGED_DIR="${MERGED_DIR:-good_results/shuf_baseline_rho}"

conda run -n cel python util/merge.py \
  --glob-shuf "__NO_SHUF_INPUT__" \
  --glob-variants "${OUTROOT}/chunk_*/baseline_rho_topology_shuffle.csv" \
  --out-dir "${MERGED_DIR}"
