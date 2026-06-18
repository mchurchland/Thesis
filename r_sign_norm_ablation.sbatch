#!/bin/bash
#SBATCH --job-name=sign_norm_ablate
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs

BASE_SEED=12345
START_SEED="${BASE_SEED}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUTROOT="sign_norm_ablation"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# Keep the full leak/input grid per task; repeat runs are partitioned by array index.
SPLIT=1
RANK=0
TOTAL_REPEATS=50
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE=100000
SID_BASE=0

# Isolation run: no neuron-bias sweep, and rho_target fixed to 1 for the requested formulas.
NEURON_BIASES=(0.0)
RHO_VALUES=(1.0)
NORMALIZATION_MODES=(spectral_radius original_radius frobenius)
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

conda run -n cel python inv_arc_test.py \
  --job "sign_test_og_cel" \
  --out-dir "${OUTDIR}" \
  --csv-name "sign_norm_ablation.csv" \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei  Connectome/ce_ei.npy \
  --src-tag "chunk_${SLURM_ARRAY_TASK_ID}" \
  --seed "${START_SEED}" \
  --sid "${SID_BASE}" \
  --split "${SPLIT}" \
  --rank "${RANK}" \
  --n-repeats "${TOTAL_REPEATS}" \
  --repeat-split "${REPEAT_SPLIT}" \
  --repeat-rank "${REPEAT_RANK}" \
  --repeat-seed-stride "${REPEAT_SEED_STRIDE}" \
  --rho-values "${RHO_VALUES[@]}" \
  --neuron-biases "${NEURON_BIASES[@]}" \
  --normalization-modes "${NORMALIZATION_MODES[@]}" \
  --ws-p 0.1 \
  --er-p 0.1 \
  --cuda "${SLURM_LOCALID:-0}" \
  --sign-flip-frac "${SIGN_FLIP_FRACS[@]}" \
  "${SUBSET_ARGS[@]}"
