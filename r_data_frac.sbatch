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

OUTROOT="sign_test"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# Keep the full parameter grid per task (repeat runs via repeat partition)
SPLIT=1
RANK=0
TOTAL_REPEATS=50
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE=100000
SID_BASE=0
NEURON_BIASES=(0.0 0.1)
CE_ADJ="${CE_ADJ:-Connectome/ce_adj_removed.npy}"
CE_EI="${CE_EI:-}"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights_removed.npy}"
SIGN_FLIP_FRACS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

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

run_fraction_sweep() {
  local label="$1"
  local job="$2"
  local unknown_policy="$3"
  local run_out_dir="${label}_${OUTDIR}"
  local unknown_args=(--unknown-sign-policy "${unknown_policy}")

  if [[ "${unknown_policy}" == "random_unknown_4to1" ]]; then
    unknown_args+=(--ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}")
  fi

  conda run -n cel python inv_arc_test.py \
    --job "${job}" \
    --out-dir "${run_out_dir}" \
    --ce-adj "${CE_ADJ}" \
    --ce-ei  "${CE_EI}" \
    --src-tag "chunk_${SLURM_ARRAY_TASK_ID}" \
    --seed "${START_SEED}" \
    --sid "${SID_BASE}" \
    --split "${SPLIT}" \
    --rank "${RANK}" \
    --n-repeats "${TOTAL_REPEATS}" \
    --repeat-split "${REPEAT_SPLIT}" \
    --repeat-rank "${REPEAT_RANK}" \
    --repeat-seed-stride "${REPEAT_SEED_STRIDE}" \
    --neuron-biases "${NEURON_BIASES[@]}" \
    "${unknown_args[@]}" \
    --ws-p 0.1 \
    --er-p 0.1 \
    --cuda "${SLURM_LOCALID:-0}" \
    --sign-flip-frac "${SIGN_FLIP_FRACS[@]}" \
    "${SUBSET_ARGS[@]}"
}

# Three new-connectome sign-fraction controls:
# 1. ER matched to the 4:1-completed new connectome edge count.
# 2. New connectome with complex/no-pred edges randomly signed 4:1 E:I per trial.
# 3. New connectome with complex/no-pred edges removed.
run_fraction_sweep "matched_er_4to1" "sign_test_er" "random_unknown_4to1"
run_fraction_sweep "new_cel_4to1" "sign_test_og_cel" "random_unknown_4to1"
run_fraction_sweep "new_cel_removed" "sign_test_og_cel" "drop"
