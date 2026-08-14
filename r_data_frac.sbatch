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

# KR+GR rank-only is the default: it preserves existing MC and IPC values.
# Use `KR_ONLY=1 RANK_ONLY=0` for KR only or set both to 0 for a full run.
KR_ONLY="${KR_ONLY:-0}"
RANK_ONLY="${RANK_ONLY:-1}"
METRIC_ARGS=()
if [[ "${KR_ONLY}" != "0" && "${KR_ONLY}" != "1" ]]; then
  echo "[error] KR_ONLY must be 0 or 1, got: ${KR_ONLY}" >&2
  exit 1
elif [[ "${RANK_ONLY}" != "0" && "${RANK_ONLY}" != "1" ]]; then
  echo "[error] RANK_ONLY must be 0 or 1, got: ${RANK_ONLY}" >&2
  exit 1
elif [[ "${KR_ONLY}" == "1" && "${RANK_ONLY}" == "1" ]]; then
  echo "[error] KR_ONLY and RANK_ONLY cannot both be 1" >&2
  exit 1
elif [[ "${KR_ONLY}" == "1" ]]; then
  METRIC_ARGS+=(--kr-only)
elif [[ "${RANK_ONLY}" == "1" ]]; then
  METRIC_ARGS+=(--rank-only)
fi

RESULTS_ROOT="${RESULTS_ROOT:-final_results_GRKR_erank}"
OUTROOT="${OUTROOT:-${RESULTS_ROOT}/sign_fraction}"
mkdir -p "${OUTROOT}"

# Keep the full parameter grid per task (repeat runs via repeat partition)
SPLIT=1
RANK=0
TOTAL_REPEATS=200
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE=100000
SID_BASE=0
NEURON_BIASES=(0.0 0.1)
CE_ADJ_REMOVED="${CE_ADJ_REMOVED:-Connectome/ce_adj.npy}"
CE_ADJ_UNREMOVED="${CE_ADJ_UNREMOVED:-Connectome/ce_adj.npy}"
CE_EI="${CE_EI:-}"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights.npy}"
SIGN_FLIP_FRACS=(0.0 0.2 0.4  0.6  0.8 1.0)

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

ensure_unknown_sign_weights() {
  if [[ -f "${CE_UNKNOWN_SIGN_WEIGHTS}" ]]; then
    return
  fi
  echo "[error] Missing processed unknown-sign weights: ${CE_UNKNOWN_SIGN_WEIGHTS}" >&2
  echo "[error] Generate/copy the processed .npy into Connectome; sbatch runs do not read XLS files." >&2
  exit 1
}

ensure_unknown_sign_weights

run_fraction_sweep() {
  local label="$1"
  local job="$2"
  local unknown_policy="$3"
  local ce_adj="$4"
  local run_out_dir="${OUTROOT}/${label}/chunk_${SLURM_ARRAY_TASK_ID}"
  local unknown_args=(--unknown-sign-policy "${unknown_policy}")

  if [[ ! -f "${ce_adj}" ]]; then
    echo "[error] Missing connectome adjacency: ${ce_adj}" >&2
    exit 1
  fi

  if [[ "${unknown_policy}" == "random_unknown_sign_matched" ]]; then
    unknown_args+=(--ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}")
  fi

  conda run -n cel python inv_arc_test.py \
    --job "${job}" \
    --out-dir "${run_out_dir}" \
    --ce-adj "${ce_adj}" \
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
    "${METRIC_ARGS[@]}" \
    "${SUBSET_ARGS[@]}"
}

# Three new-connectome sign-fraction controls:
# 1. ER matched to the sign-matched completed connectome edge count.
# 2. New connectome with complex/no-pred edges signed at the observed CE rate per trial.
# 3. New connectome with complex/no-pred edges removed.
run_fraction_sweep "matched_er_sign_matched" "sign_test_er" "random_unknown_sign_matched" "${CE_ADJ_UNREMOVED}"
run_fraction_sweep "new_cel_sign_matched" "sign_test_og_cel" "random_unknown_sign_matched" "${CE_ADJ_UNREMOVED}"
run_fraction_sweep "new_cel_removed" "sign_test_og_cel" "drop" "${CE_ADJ_REMOVED}"
