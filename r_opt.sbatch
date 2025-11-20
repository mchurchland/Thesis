#!/bin/bash
#SBATCH --job-name=ce_eanomad_opt
#SBATCH --partition=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# Uncomment to run multiple independent seeds:
# SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs

BASE_SEED=12345
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( BASE_SEED + TASK_ID * 100000 ))

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
  --mode rEA \
  --generations 80 \
  --population-size 32 \
  --subset-size 20 \
  --bounds 0.1 \
  --max-bb-eval 150 \
  --n-mutate-coords 5 \
  --rho-targets 0.9 1.0 1.1 \
  --leaks 0.2 0.5 0.8 \
  --input-scales 0.5 1.0 2.0 \
  --w-mc 1 --w-ipc 1 --w-kr 1 --w-gr 1 \
  --norm ratio \
  --l2-penalty 0.0 \
  --seed "${SEED}"
