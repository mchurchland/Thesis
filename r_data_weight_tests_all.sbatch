#!/bin/bash
#SBATCH --job-name=ce_weight_tests
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-31             # 32 total array tasks across all weight-test variants
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs

BASE_SEED=12345
START_SEED="${BASE_SEED}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

VARIANTS=(
  "weight_test_unsigned"
  "weight_test_signed"
  "weight_test_cel_to_shuffled_cel"
  "weight_test_binary_to_cel"
  "weight_test_binary_to_shuffled_cel"
)

N_VARIANTS=${#VARIANTS[@]}
ARRAY_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
ARRAY_MAX="${SLURM_ARRAY_TASK_MAX:-31}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_OFFSET=$(( TASK_ID - ARRAY_MIN ))
TOTAL_ARRAY_TASKS=$(( ARRAY_MAX - ARRAY_MIN + 1 ))

if (( TOTAL_ARRAY_TASKS < N_VARIANTS )); then
  echo "Array size ${TOTAL_ARRAY_TASKS} must be at least ${N_VARIANTS} variants." >&2
  exit 1
fi

VARIANT_IDX=$(( TASK_OFFSET % N_VARIANTS ))
REPEAT_RANK=$(( TASK_OFFSET / N_VARIANTS ))
REPEAT_SPLIT=$(( (TOTAL_ARRAY_TASKS + N_VARIANTS - 1 - VARIANT_IDX) / N_VARIANTS ))
JOB_KIND="${VARIANTS[VARIANT_IDX]}"

if [[ "${JOB_KIND}" == weight_test_binary_to_* || "${JOB_KIND}" == "weight_test_cel_to_shuffled_cel" ]]; then
  ALPHAS=(0.0 0.25 0.5 0.75 1.0)
else
  ALPHAS=(0.0 1.0 5.0 10.0)
fi

OUTROOT="final_weight_tests_all"
OUTDIR="${OUTROOT}/${JOB_KIND}/chunk_${REPEAT_RANK}"
mkdir -p "${OUTDIR}"

# Keep the full parameter grid per task; repeat runs are partitioned by array index.
SPLIT=1
RANK=0
TOTAL_REPEATS=250
REPEAT_SEED_STRIDE=100000
SID_BASE=0

RANDOM_NODE_SUBSETS=0
K_IN=""
K_OUT=""
SUBSET_SEED_OFFSET=""

SUBSET_ARGS=()
if [[ "${RANDOM_NODE_SUBSETS}" == "1" ]]; then
  : "${K_IN:?Set K_IN when RANDOM_NODE_SUBSETS=1}"
  : "${K_OUT:?Set K_OUT when RANDOM_NODE_SUBSETS=1}"
  SUBSET_ARGS=(--random-node-subsets --k-in "${K_IN}" --k-out "${K_OUT}")
  if [[ -n "${SUBSET_SEED_OFFSET}" ]]; then
    SUBSET_ARGS+=(--subset-seed-offset "${SUBSET_SEED_OFFSET}")
  fi
fi

conda run -n cel python inv_arc_test.py \
  --job "${JOB_KIND}" \
  --out-dir "${OUTDIR}" \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei  Connectome/ce_ei.npy \
  --src-tag "${JOB_KIND}_chunk_${REPEAT_RANK}" \
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
  --cuda "${SLURM_LOCALID:-0}" \
  --alphas "${ALPHAS[@]}" \
  "${SUBSET_ARGS[@]}"
