#!/bin/bash
#SBATCH --job-name=inv_over_ei
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=1-20
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs

# Ensure we run from the directory where sbatch was submitted
cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- paths (edit as needed) ---
CE_ADJ="Connectome/ce_adj.npy"
CE_EI="Connectome/ce_ei.npy"
OUTROOT="inv_over_ei_chunks"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# total number of array tasks -> used for --split
SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

# --- RNG / device ---
BASE_SEED=12345
START_SEED=$(( BASE_SEED + SLURM_ARRAY_TASK_ID * 100000 ))
CUDA_IDX=0  # set to -1 to force CPU

# Activate your environment and run

conda run -n cel python inv_over_ei.py \
 --ce-adj "${CE_ADJ}" \
 --ce-ei "${CE_EI}" \
 --out-dir "${OUTDIR}" \
 --seed "${START_SEED}" \
 --n-seeds 25 \
 --seed-stride 101 \
 --split "${SPLIT}" \
 --rank "${SLURM_ARRAY_TASK_ID}" \
 --out-tag "chunk_${SLURM_ARRAY_TASK_ID}" \
 --no-plot \
 --cuda "${CUDA_IDX}"

echo "Done."
