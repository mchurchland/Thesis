#!/bin/bash
#SBATCH --job-name=ce_eanomad_opt
#SBATCH --partition=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Uncomment to run multiple independent seeds:
# SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

BASE_SEED=12345
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

OUTROOT="eanomad_ce_mc_ipc_kr_gr_runs"
OUTDIR="${OUTROOT}/seed_${TASK_ID}"
mkdir -p "${OUTDIR}"

conda run -n cel python opt_cel.py \
  --ce-adj Connectome/ce_adj.npy \
  --ce-ei  Connectome/ce_ei.npy \
  --out-dir "${OUTDIR}" \
  --device cuda \
  --mode EA \
  --generations 2 \
  --population-size 4 \
  --subset-size 49 \
  --bounds 1 \
  --max-bb-eval 20 \
  --n-mutate-coords 0 \
  --rho-targets 0.99 \
  --leaks  0.95 \
  --input-scales 1.0 \
  --w-mc 1 --w-ipc 1 --w-kr 1\
  --w-gr -1 \
  --l2-penalty 0.0 \
  --seed "${SEED}"\
  --n-seeds-per-eval 4