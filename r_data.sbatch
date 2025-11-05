#!/bin/bash
#SBATCH --job-name=ce_1k_each
#SBATCH --partition=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs

# 20 array tasks × 50 reps/task = 1000 per architecture
REPS_PER_TASK=50

# isolate seeds per chunk to avoid collisions
BASE_SEED=12345
START_SEED=$(( BASE_SEED + SLURM_ARRAY_TASK_ID * 100000 ))

# bind BLAS threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUTROOT="experiment_full_chunks"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

conda run -n cel python dist_exp.py \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei  Connectome/ce_ei.npy \
  --out-dir "${OUTDIR}" \
  --src-tag "chunk_${SLURM_ARRAY_TASK_ID}" \
  --device cuda \
  --threads ${SLURM_CPUS_PER_TASK} \
  --seed ${START_SEED} \
  --n-ce-real-reps ${REPS_PER_TASK} \
  --n-rand-reps    ${REPS_PER_TASK} \
  --er-p 0.1 \
  --ws-p 0.1 \
  --ce-real-csv ce_real_reps.csv \
  --variants-csv invariance_variants.csv
