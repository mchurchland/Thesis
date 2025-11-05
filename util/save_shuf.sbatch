#!/bin/bash
#SBATCH --job-name=filter_shuf
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run the filter script on the given file
conda run -n cel python save_shuf.py \
  --input  /home/milesc/Thesis/experiment_full_merged.v1/shuf+connshuf.csv \
  --output /home/milesc/Thesis/experiment_full_merged.v1/filtered_shuffles.csv
