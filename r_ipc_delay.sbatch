#!/bin/bash
#SBATCH --job-name=ipc_delay
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
OUT_DIR="ipc_delay_sensitivity_all"

# Sweep settings
D_MAX_LIST="30 50 80"
K_ORDERS="1 3 5"
N_SEEDS=20
SEED_START=0

CUDA_IDX="${SLURM_LOCALID:-0}"

conda run -n cel python ipc_delay_sensitivity.py \
  --ce-adj "${CE_ADJ}" \
  --ce-ei "${CE_EI}" \
  --out-dir "${OUT_DIR}" \
  --models all \
  --n-seeds "${N_SEEDS}" \
  --seed-start "${SEED_START}" \
  --k-orders ${K_ORDERS} \
  --d-max-list ${D_MAX_LIST} \
  --cuda "${CUDA_IDX}"
