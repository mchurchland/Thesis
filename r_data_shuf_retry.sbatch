#!/bin/bash
#SBATCH --job-name=ce_shuf_retry
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=18
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# These task IDs came from the original 0-19 array. Keep its 20-way repeat
# partition even though this retry is a sparse four-task array.
export REPEAT_SPLIT=20

# This retry is intended to finish the KR/GR run. r_data_shuf.sbatch defaults
# to rank-only and preserves the existing MC and IPC columns.
export KR_ONLY="${KR_ONLY:-0}"
export RANK_ONLY="${RANK_ONLY:-1}"

exec bash r_data_shuf.sbatch
