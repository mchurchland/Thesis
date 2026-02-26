#!/bin/bash
#SBATCH --job-name=reservoir_run
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs1/%x-%j.out
#SBATCH --error=logs1/%x-%j.err

set -euo pipefail

mkdir -p logs1

# Ensure we run from the directory where sbatch was submitted
cd "${SLURM_SUBMIT_DIR}"

# Avoid BLAS oversubscription; CPU-only job
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Activate env and run script that calls run_reservoir_with_pre(...)
conda run -n cel python normality_test.py
