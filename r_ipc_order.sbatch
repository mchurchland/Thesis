#!/bin/bash
#SBATCH --job-name=ipc_order
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-ipc_order}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RESULTS_ROOT="${RESULTS_ROOT:-final_results}"
CE_ADJ="${CE_ADJ:-Connectome/ce_adj.npy}"
CE_EI=""
UNKNOWN_SIGN_POLICY="random_unknown_sign_matched"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights.npy}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/ipc_order}"

# Sweep settings
N_SEEDS=20
SEED_START=0
MAX_DELAY=50
MAX_ORDER=10
NEURON_BIASES=(0.0)
CAPACITY_THRESHOLDS=(0.0)

CUDA_IDX="${SLURM_LOCALID:-0}"

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

conda run -n cel python Parameter_justification/ipc_order_sweep_ce.py \
  --ce-adj "${CE_ADJ}" \
  --ce-ei "${CE_EI}" \
  --out-dir "${OUT_DIR}" \
  --models all \
  --n-seeds "${N_SEEDS}" \
  --seed-start "${SEED_START}" \
  --max-delay "${MAX_DELAY}" \
  --max-order "${MAX_ORDER}" \
  --neuron-biases "${NEURON_BIASES[@]}" \
  --capacity-thresholds "${CAPACITY_THRESHOLDS[@]}" \
  "${UNKNOWN_SIGN_ARGS[@]}" \
  --cuda "${CUDA_IDX}"
