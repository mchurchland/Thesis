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
TOTAL_REPEATS=150
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE=100000
SID_BASE=0
NEURON_BIASES=(0.0 0.1)
CE_ADJ="${CE_ADJ:-Connectome/ce_adj_removed.npy}"
CE_EI="${CE_EI:-}"
UNKNOWN_SIGN_POLICY="${UNKNOWN_SIGN_POLICY:-random_unknown_4to1}"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights_removed.npy}"
NEW_CEL_XLS="${NEW_CEL_XLS:-new_cel.xls}"
OUT_PREFIX="${OUT_PREFIX:-shuf_sep_pm1_fix_new_cel_${UNKNOWN_SIGN_POLICY}}"

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
  if [[ "${UNKNOWN_SIGN_POLICY}" != "random_unknown_4to1" ]]; then
    return
  fi
  if [[ -f "${CE_UNKNOWN_SIGN_WEIGHTS}" ]]; then
    return
  fi
  if [[ ! -f "${NEW_CEL_XLS}" ]]; then
    echo "[error] Missing ${CE_UNKNOWN_SIGN_WEIGHTS} and cannot regenerate it because ${NEW_CEL_XLS} is missing." >&2
    exit 1
  fi

  local tmp_prefix="${TMPDIR:-/tmp}/ce_unknown_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
  echo "[info] ${CE_UNKNOWN_SIGN_WEIGHTS} not found; regenerating task-local unknown-sign weights from ${NEW_CEL_XLS}"
  conda run -n cel python util/read_xls.py \
    --xls "${NEW_CEL_XLS}" \
    --new-cel \
    --out "${tmp_prefix}"
  CE_UNKNOWN_SIGN_WEIGHTS="${tmp_prefix}_unknown_sign_weights.npy"
  echo "[info] using generated unknown-sign weights: ${CE_UNKNOWN_SIGN_WEIGHTS}"
}

ensure_unknown_sign_weights

UNKNOWN_SIGN_ARGS=(--unknown-sign-policy "${UNKNOWN_SIGN_POLICY}")
if [[ "${UNKNOWN_SIGN_POLICY}" == "random_unknown_4to1" ]]; then
  UNKNOWN_SIGN_ARGS+=(--ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}")
fi

# choose which variant this script runs
JOB_KIND="all_topology_shuffle" 

conda run -n cel python inv_arc_test.py \
  --job "${JOB_KIND}" \
  --out-dir "${OUT_PREFIX}_${OUTDIR}" \
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
  "${UNKNOWN_SIGN_ARGS[@]}" \
  --ws-p 0.1 \
  --er-p 0.1 \
  --cuda ${SLURM_LOCALID} \
  "${SUBSET_ARGS[@]}"
