#!/bin/bash
#SBATCH --job-name=ipc_method
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-ipc_method}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Focused IPC methodology check for the new connectome and bias experiment.
# ce_adj_unk41.npy has no matching EI file in this repo; passing an empty EI path
# lets load_connectome use the adjacency without trying to load old-size EI labels.
CE_ADJ="${CE_ADJ:-Connectome/ce_adj_unk41.npy}"
CE_EI=""
UNKNOWN_SIGN_POLICY="${UNKNOWN_SIGN_POLICY:-random_unknown_4to1}"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights_unk41.npy}"
BASE_OUT_DIR="${BASE_OUT_DIR:-ipc_methodology_new_connectome_${UNKNOWN_SIGN_POLICY}}"

# Representative model/hyperparameter set. This is intentionally not the full
# architecture sweep.
MODELS=(real global_sign_pres_real_weight binary_base er_randN)
RHO_VALUES=(0.95 1.05)
LEAK_VALUES=(0.8)
INPUT_SCALES=(1.0)

# Methodology checks requested by Miguel.
NEURON_BIASES=(0.0 0.1)
CAPACITY_THRESHOLDS=(0.0 0.001 0.005 0.01)

N_SEEDS=10
SEED_START=0
MAX_DELAY=80
MAX_ORDER=10
WASHOUT=500
SERIES_SETTINGS=(
  "default:1500:500"
  "long:6000:2000"
)
RIDGE_ALPHA=1e-4

CUDA_IDX="${SLURM_LOCALID:-0}"

UNKNOWN_SIGN_ARGS=(--unknown-sign-policy "${UNKNOWN_SIGN_POLICY}")
if [[ "${UNKNOWN_SIGN_POLICY}" == "random_unknown_4to1" ]]; then
  if [[ ! -f "${CE_UNKNOWN_SIGN_WEIGHTS}" ]]; then
    echo "[error] Missing processed unknown-sign weights: ${CE_UNKNOWN_SIGN_WEIGHTS}" >&2
    echo "[error] Generate/copy the processed .npy into Connectome; sbatch runs do not read XLS files." >&2
    exit 1
  fi
  UNKNOWN_SIGN_ARGS+=(--ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}")
fi

for setting in "${SERIES_SETTINGS[@]}"; do
  IFS=: read -r SERIES_LABEL T_TRAIN T_TEST <<< "${setting}"
  OUT_DIR="${BASE_OUT_DIR}/${SERIES_LABEL}"

  conda run -n cel python Parameter_justification/ipc_order_sweep_ce.py \
    --ce-adj "${CE_ADJ}" \
    --ce-ei "${CE_EI}" \
    --out-dir "${OUT_DIR}" \
    --models "${MODELS[@]}" \
    --n-seeds "${N_SEEDS}" \
    --seed-start "${SEED_START}" \
    --rho "${RHO_VALUES[@]}" \
    --leak "${LEAK_VALUES[@]}" \
    --input-scale "${INPUT_SCALES[@]}" \
    --neuron-biases "${NEURON_BIASES[@]}" \
    --capacity-thresholds "${CAPACITY_THRESHOLDS[@]}" \
    "${UNKNOWN_SIGN_ARGS[@]}" \
    --washout "${WASHOUT}" \
    --t-train "${T_TRAIN}" \
    --t-test "${T_TEST}" \
    --max-delay "${MAX_DELAY}" \
    --max-order "${MAX_ORDER}" \
    --ridge-alpha "${RIDGE_ALPHA}" \
    --cuda "${CUDA_IDX}"
done
