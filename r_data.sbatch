#!/bin/bash
#SBATCH --job-name=ce_real_inv
#SBATCH --partition=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=1              # 20 chunks of the param grid
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs


BASE_SEED=12345
START_SEED=$(( BASE_SEED + SLURM_ARRAY_TASK_ID * 100000 ))

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUTROOT="experiment_full_chunks"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# total number of array tasks -> used for --split
SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

# choose which variant this script runs
JOB_KIND="real"   # or shuffle_weights / cel_randN / er_randN / ws_p01_randN / conn_shuf

conda run -n cel python inv_arc_test.py \
  --job "${JOB_KIND}" \
  --out-dir "${OUTDIR}" \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei  Connectome/ce_ei.npy \
  --src-tag "chunk_${SLURM_ARRAY_TASK_ID}" \
  --seed "${START_SEED}" \
  --sid "${SLURM_ARRAY_TASK_ID}" \
  --split "${SPLIT}" \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --ws-p 0.1 \
  --er-p 0.1 \
  --cuda 0

