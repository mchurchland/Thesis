#!/bin/bash
#SBATCH --job-name=time_window_mc
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- output layout ---
OUTROOT="time_window_sensitivity_chunks"
OUTDIR="${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUTDIR}"

# --- seed partitioning across array tasks ---
BASE_SEED=0
SEEDS_PER_TASK=1
SEED_START=$(( BASE_SEED + SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))

# --- core run controls ---
MODELS="all"
METRIC="mc"
MAX_DELAY=30
WASHOUT=500
T_TRAIN=1500
T_TEST=500
RIDGE_ALPHA=1e-4
SCALE_FACTORS="0.25 0.75 1.0 1.25"

# Use the same default hyperparameter grids as inv_arc_test.py.
RHO_GRID="0.6 0.8 0.95 1.05"
LEAK_GRID="0.6 0.8 1.0"
INPUT_SCALE_GRID="0.1 0.5 1.0 1.5"

# Set to >=0 if you want a specific CUDA index per task.
CUDA_IDX=-1

CMD=(
  conda run -n cel python time_window_sensitivity.py
  --out-dir "${OUTDIR}"
  --models ${MODELS}
  --seed-start "${SEED_START}"
  --n-seeds "${SEEDS_PER_TASK}"
  --metric "${METRIC}"
  --max-delay "${MAX_DELAY}"
  --washout "${WASHOUT}"
  --t-train "${T_TRAIN}"
  --t-test "${T_TEST}"
  --ridge-alpha "${RIDGE_ALPHA}"
  --scale-factors ${SCALE_FACTORS}
  --rho ${RHO_GRID}
  --leak ${LEAK_GRID}
  --input-scale ${INPUT_SCALE_GRID}
  --parallel-backend none
)

if [[ "${CUDA_IDX}" -ge 0 ]]; then
  CMD+=(--cuda "${CUDA_IDX}")
fi

"${CMD[@]}"

echo "Done. Wrote ${OUTDIR}"
