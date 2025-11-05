#!/bin/bash
#SBATCH --job-name=ce_graph
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# BLAS threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run graphing.py
conda run -n cel python graph_hist.py \
  --combined experiment_full_merged.v2/combined.ALL.csv \
  --out-dir experiment_full_merged.v2 \
  --bins 100 \
  --scatter-alpha 0.55