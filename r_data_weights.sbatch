#!/bin/bash
#SBATCH --job-name=ce_real_inv
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19             # 20 array tasks; repeats partitioned in code
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs


BASE_SEED=12345
START_SEED="${BASE_SEED}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUTROOT="experiment_full_chunks"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# Keep the full parameter grid per task (repeat runs via repeat partition)
SPLIT=1
RANK=0
TOTAL_REPEATS=250
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE=100000
SID_BASE=0

# choose which variant this script runs
JOB_KIND="weight_test"   # or all / local_sign+sample/ local_sign+flat / shuffle_weights / cel_randN / er_randN / ws_p01_randN / conn_shuf / local_sign / real

conda run -n cel python inv_arc_test.py \
  --job "${JOB_KIND}" \
  --out-dir "final_weights_${OUTDIR}" \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei  Connectome/ce_ei.npy \
  --src-tag "chunk_${SLURM_ARRAY_TASK_ID}" \
  --seed "${START_SEED}" \
  --sid "${SID_BASE}" \
  --split "${SPLIT}" \
  --rank "${RANK}" \
  --n-repeats "${TOTAL_REPEATS}" \
  --repeat-split "${REPEAT_SPLIT}" \
  --repeat-rank "${REPEAT_RANK}" \
  --repeat-seed-stride "${REPEAT_SEED_STRIDE}" \
  --ws-p 0.1 \
  --er-p 0.1 \
  --cuda ${SLURM_LOCALID}\
  --alpha 0.0 1.0 5.0 10.0