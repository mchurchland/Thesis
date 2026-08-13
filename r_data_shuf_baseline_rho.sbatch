#!/bin/bash
#SBATCH --job-name=shuf_base_rho
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs

BASE_SEED=12345
START_SEED="${BASE_SEED}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Submit with `RANK_ONLY=1 sbatch r_data_shuf_baseline_rho.sbatch` to update only KR/GR.
RANK_ONLY="${RANK_ONLY:-1}"
METRIC_ARGS=()
if [[ "${RANK_ONLY}" == "1" ]]; then
  METRIC_ARGS+=(--rank-only)
elif [[ "${RANK_ONLY}" != "0" ]]; then
  echo "[error] RANK_ONLY must be 0 or 1, got: ${RANK_ONLY}" >&2
  exit 1
fi

RESULTS_ROOT="${RESULTS_ROOT:-final_results_GRKR}"
OUTROOT="${OUTROOT:-${RESULTS_ROOT}/baseline_rho_topology_shuffles}"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# Keep the full hyperparameter grid per task; repeat runs are partitioned by array index.
SPLIT=1
RANK=0
TOTAL_REPEATS="${TOTAL_REPEATS:-200}"
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE=100000
SID_BASE=0
NEURON_BIASES=(0.0 0.1)

CE_ADJ="${CE_ADJ:-Connectome/ce_adj.npy}"
CE_EI="${CE_EI:-}"
UNKNOWN_SIGN_POLICY="${UNKNOWN_SIGN_POLICY:-random_unknown_sign_matched}"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights.npy}"

RANDOM_NODE_SUBSETS="${RANDOM_NODE_SUBSETS:-0}"
K_IN="${K_IN:-}"
K_OUT="${K_OUT:-}"
SUBSET_SEED_OFFSET="${SUBSET_SEED_OFFSET:-}"

SUBSET_ARGS=()
if [[ "${RANDOM_NODE_SUBSETS}" == "1" ]]; then
  : "${K_IN:?Set K_IN when RANDOM_NODE_SUBSETS=1}"
  : "${K_OUT:?Set K_OUT when RANDOM_NODE_SUBSETS=1}"
  SUBSET_ARGS=(--random-node-subsets --k-in "${K_IN}" --k-out "${K_OUT}")
  if [[ -n "${SUBSET_SEED_OFFSET}" ]]; then
    SUBSET_ARGS+=(--subset-seed-offset "${SUBSET_SEED_OFFSET}")
  fi
fi

if [[ ! -f "${CE_ADJ}" ]]; then
  echo "[error] Missing adjacency: ${CE_ADJ}" >&2
  exit 1
fi
if [[ "${UNKNOWN_SIGN_POLICY}" != "random_unknown_sign_matched" ]]; then
  echo "[error] This baseline-rho shuffle experiment expects UNKNOWN_SIGN_POLICY=random_unknown_sign_matched." >&2
  exit 1
fi
if [[ ! -f "${CE_UNKNOWN_SIGN_WEIGHTS}" ]]; then
  echo "[error] Missing unknown-sign weights: ${CE_UNKNOWN_SIGN_WEIGHTS}" >&2
  exit 1
fi

UNKNOWN_SIGN_ARGS=(--unknown-sign-policy "${UNKNOWN_SIGN_POLICY}")
UNKNOWN_SIGN_ARGS+=(--ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}")

conda run -n cel python inv_arc_test.py \
  --job all_topology_shuffle \
  --out-dir "${OUTDIR}" \
  --csv-name "baseline_rho_topology_shuffle.csv" \
  --ce-adj "${CE_ADJ}" \
  --ce-ei "${CE_EI}" \
  --src-tag "baseline_rho_chunk_${SLURM_ARRAY_TASK_ID}" \
  --seed "${START_SEED}" \
  --sid "${SID_BASE}" \
  --split "${SPLIT}" \
  --rank "${RANK}" \
  --n-repeats "${TOTAL_REPEATS}" \
  --repeat-split "${REPEAT_SPLIT}" \
  --repeat-rank "${REPEAT_RANK}" \
  --repeat-seed-stride "${REPEAT_SEED_STRIDE}" \
  --neuron-biases "${NEURON_BIASES[@]}" \
  --normalization-modes original_radius \
  "${UNKNOWN_SIGN_ARGS[@]}" \
  --ws-p 0.1 \
  --er-p 0.1 \
  --cuda "${SLURM_LOCALID:-0}" \
  "${METRIC_ARGS[@]}" \
  "${SUBSET_ARGS[@]}"
