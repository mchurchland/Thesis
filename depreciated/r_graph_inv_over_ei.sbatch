#!/bin/bash
#SBATCH --job-name=inv_over_ei
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CE_ADJ="Connectome/ce_adj.npy"
CE_EI="Connectome/ce_ei.npy"
OUTDIR="ladder_results"

BASE_SEED=12345
CUDA_IDX=0

conda run -n cel python inv_over_ei.py \
  --plot-only \
  --summary-glob "inv_over_ei_chunks/chunk_*/ladder_summary__chunk_*.csv" \
  --out-dir "ladder_results"



echo "Done."
