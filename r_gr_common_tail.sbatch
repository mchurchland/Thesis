#!/bin/bash
#SBATCH --job-name=gr_tail
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p logs
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID:-gr_tail}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RESULTS_ROOT="${RESULTS_ROOT:-final_results}"
RUN_TAG="${RUN_TAG:-job_${SLURM_JOB_ID:-manual}}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/gr_common_tail_sensitivity/${RUN_TAG}}"

CE_ADJ="${CE_ADJ:-Connectome/ce_adj.npy}"
CE_EI="${CE_EI:-}"
UNKNOWN_SIGN_POLICY="${UNKNOWN_SIGN_POLICY:-random_unknown_sign_matched}"
CE_UNKNOWN_SIGN_WEIGHTS="${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights.npy}"

# The current production protocol is 300 streams, total length 10, and a
# three-step common tail. Lengths 0 and 10 are limiting controls; interpret
# robustness from the interior lengths rather than treating the controls as
# ordinary GR conditions.
NUM_STREAMS="${NUM_STREAMS:-300}"
STREAM_LENGTH="${STREAM_LENGTH:-10}"
REFERENCE_TAIL_LENGTH="${REFERENCE_TAIL_LENGTH:-3}"
COMMON_TAIL_LENGTHS_TEXT="${COMMON_TAIL_LENGTHS:-0 1 2 3 4 5 7 10}"
read -r -a COMMON_TAIL_LENGTH_ARGS <<< "${COMMON_TAIL_LENGTHS_TEXT}"

# Focused parameter-justification settings. Override these strings with
# sbatch --export or edit them to cover a broader hyperparameter grid.
MODELS_TEXT="${MODELS:-all}"
RHO_VALUES_TEXT="${RHO_VALUES:-0.95}"
LEAK_VALUES_TEXT="${LEAK_VALUES:-0.8}"
INPUT_SCALES_TEXT="${INPUT_SCALES:-1.0}"
NEURON_BIASES_TEXT="${NEURON_BIASES:-0.0}"
read -r -a MODEL_ARGS <<< "${MODELS_TEXT}"
read -r -a RHO_ARGS <<< "${RHO_VALUES_TEXT}"
read -r -a LEAK_ARGS <<< "${LEAK_VALUES_TEXT}"
read -r -a INPUT_SCALE_ARGS <<< "${INPUT_SCALES_TEXT}"
read -r -a NEURON_BIAS_ARGS <<< "${NEURON_BIASES_TEXT}"

N_SEEDS="${N_SEEDS:-20}"
SEED_START="${SEED_START:-0}"
CUDA_IDX="${SLURM_LOCALID:-0}"

if [[ ! -f "${CE_ADJ}" ]]; then
  echo "[error] Missing connectome adjacency: ${CE_ADJ}" >&2
  exit 1
fi

UNKNOWN_SIGN_ARGS=(--unknown-sign-policy "${UNKNOWN_SIGN_POLICY}")
if [[ "${UNKNOWN_SIGN_POLICY}" == "random_unknown_sign_matched" ]]; then
  if [[ ! -f "${CE_UNKNOWN_SIGN_WEIGHTS}" ]]; then
    echo "[error] Missing unknown-sign weights: ${CE_UNKNOWN_SIGN_WEIGHTS}" >&2
    exit 1
  fi
  UNKNOWN_SIGN_ARGS+=(--ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS}")
fi

mkdir -p "${OUT_DIR}"
echo "[info] GR common-tail sweep output: ${OUT_DIR}"
echo "[info] tail lengths: ${COMMON_TAIL_LENGTH_ARGS[*]} (reference=${REFERENCE_TAIL_LENGTH})"

conda run -n cel python Parameter_justification/gr_common_tail_sensitivity.py \
  --ce-adj "${CE_ADJ}" \
  --ce-ei "${CE_EI}" \
  --out-dir "${OUT_DIR}" \
  --models "${MODEL_ARGS[@]}" \
  --n-seeds "${N_SEEDS}" \
  --seed-start "${SEED_START}" \
  --rho "${RHO_ARGS[@]}" \
  --leak "${LEAK_ARGS[@]}" \
  --input-scale "${INPUT_SCALE_ARGS[@]}" \
  --neuron-biases "${NEURON_BIAS_ARGS[@]}" \
  --num-streams "${NUM_STREAMS}" \
  --stream-length "${STREAM_LENGTH}" \
  --common-tail-lengths "${COMMON_TAIL_LENGTH_ARGS[@]}" \
  --reference-tail-length "${REFERENCE_TAIL_LENGTH}" \
  "${UNKNOWN_SIGN_ARGS[@]}" \
  --cuda "${CUDA_IDX}"
