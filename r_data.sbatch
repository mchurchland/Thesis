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

RESULTS_ROOT="${RESULTS_ROOT:-final_results}"
OUTROOT="${OUTROOT:-${RESULTS_ROOT}/main_experiment}"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# Keep the full parameter grid per task (repeat runs via repeat partition)
SPLIT=1
RANK=0
TOTAL_REPEATS=200
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE=100000
SID_BASE=0
NEURON_BIASES=(0.0 0.1)
CE_ADJ="${CE_ADJ:-Connectome/ce_adj.npy}"
CE_EI="${CE_EI:-}"
UNKNOWN_SIGN_POLICY="random_unknown_sign_matched"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights.npy}"

RANDOM_NODE_SUBSETS=0
K_IN=""
K_OUT="" ## for 20 %
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

for required_file in "${CE_ADJ}" "${CE_UNKNOWN_SIGN_WEIGHTS}"
do
  if [[ ! -f "${required_file}" ]]; then
    echo "[error] Missing required sign-matched input: ${required_file}" >&2
    exit 1
  fi
done

UNKNOWN_SIGN_ARGS=(
  --unknown-sign-policy "${UNKNOWN_SIGN_POLICY}"
  --ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}"
)

# choose which variant this script runs
JOB_KIND="all"   # or all / global_sign_pres_real_weight / local_sign+binary / local_sign+sample / local_sign+flat / shuffle_weights / cel_randN / er_randN / ws_p01_randN / conn_shuf / local_sign / real

conda run -n cel python inv_arc_test.py \
  --job "${JOB_KIND}" \
  --out-dir "${OUTDIR}" \
  --ce-adj "${CE_ADJ}" \
  --ce-ei "${CE_EI}" \
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
  --cuda "${SLURM_LOCALID:-0}" \
  "${SUBSET_ARGS[@]}"
