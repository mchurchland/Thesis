#!/bin/bash
#SBATCH --job-name=mc_delay
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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

CE_ADJ="Connectome/ce_adj.npy"
CE_EI="Connectome/ce_ei.npy"
UNKNOWN_SIGN_POLICY="${UNKNOWN_SIGN_POLICY:-drop}"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-}"
OUT_DIR="mc_delay_sensitivity_all"

# Sweep settings
D_MAX_LIST="30 50 80"
N_SEEDS=20
SEED_START=0

CUDA_IDX="${SLURM_LOCALID:-0}"

UNKNOWN_SIGN_ARGS=(--unknown-sign-policy "${UNKNOWN_SIGN_POLICY}")
if [[ "${UNKNOWN_SIGN_POLICY}" == "random_unknown_4to1" && -n "${CE_UNKNOWN_SIGN_WEIGHTS}" ]]; then
  UNKNOWN_SIGN_ARGS+=(--ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}")
fi

conda run -n cel python Parameter_justification/mc_delay_sensitivity.py \
  --ce-adj "${CE_ADJ}" \
  --ce-ei "${CE_EI}" \
  --out-dir "${OUT_DIR}" \
  --models all \
  --n-seeds "${N_SEEDS}" \
  --seed-start "${SEED_START}" \
  --d-max-list ${D_MAX_LIST} \
  "${UNKNOWN_SIGN_ARGS[@]}" \
  --cuda "${CUDA_IDX}"
