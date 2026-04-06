#!/bin/bash
#SBATCH --job-name=ladder
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# --- paths (edit as needed) ---
CE_ADJ="Connectome/ce_adj.npy"
CE_EI="Connectome/ce_ei.npy"
OUT_DIR="ladder_results"

# --- sweep controls ---
FRACTIONS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
WS_K=40
N_SEEDS=15
SEED_STRIDE=101

# --- RNG / device ---
BASE_SEED=12345
CUDA_IDX=0  # set to -1 to force CPU

# Activate your environment and run
conda run -n cel python weight_destruction_ladder_sanity.py \
  --ce-adj "${CE_ADJ}" \
  --ce-ei  "${CE_EI}" \
  --out-dir "${OUT_DIR}" \
  --fractions ${FRACTIONS} \
  --ws-k ${WS_K} \
  --seed ${BASE_SEED} \
  --n-seeds ${N_SEEDS} \
  --seed-stride ${SEED_STRIDE} \
  --cuda ${CUDA_IDX}

echo "Done."
 # --rho-targets ${RHO_TARGETS} \
 # --leaks ${LEAKS} \
 # --input-scales ${INPUT_SCALES} \