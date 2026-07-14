#!/bin/bash
#SBATCH --job-name=triad_sign
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RESULTS_ROOT="${RESULTS_ROOT:-final_results}"
OUTDIR="${OUTDIR:-${RESULTS_ROOT}/triad_sign_fraction/chunk_${SLURM_ARRAY_TASK_ID}}"
BASE_SEED="${BASE_SEED:-12345}"
TOTAL_REPEATS="${TOTAL_REPEATS:-50}"
REPEAT_SPLIT=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
REPEAT_RANK="${SLURM_ARRAY_TASK_ID}"
REPEAT_SEED_STRIDE="${REPEAT_SEED_STRIDE:-100000}"

SIGN_FRACS=(0.0 0.5)
RHO_VALUES=(0.6 0.8 0.95 1.05)
NORMALIZATION_MODES=(spectral_radius)

conda run -n cel python network_stats/triad_sign_fraction_experiment.py \
  --out-dir "${OUTDIR}" \
  --seed "${BASE_SEED}" \
  --n-repeats "${TOTAL_REPEATS}" \
  --repeat-split "${REPEAT_SPLIT}" \
  --repeat-rank "${REPEAT_RANK}" \
  --repeat-seed-stride "${REPEAT_SEED_STRIDE}" \
  --datasets new_cel_sign_matched \
  --sign-fracs "${SIGN_FRACS[@]}" \
  --no-include-empirical-cel-fraction \
  --rho-values "${RHO_VALUES[@]}" \
  --normalization-modes "${NORMALIZATION_MODES[@]}"
