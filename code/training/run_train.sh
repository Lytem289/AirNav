#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash training/run_train.sh GENERATED_DIR [LLAMAFACTORY_DIR] [smoke|336|448] [RUN_NAME] [auto|thought_action|action_only|decision_action] [RESUME_CHECKPOINT]"
  exit 2
fi

GENERATED_DIR="$1"
LLAMAFACTORY_DIR="${2:-../LLaMA-Factory}"
PROFILE="${3:-smoke}"
RUN_NAME="${4:-}"
ASSISTANT_FORMAT="${5:-auto}"
RESUME_CHECKPOINT="${6:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "${PROFILE}" in
  smoke) CONFIG="${SCRIPT_DIR}/qwen2_5vl_lora_sft_smoke.yaml"; DEFAULT_OUTPUT="saves/qwen2_5vl-7b/lora/uav_sft_smoke" ;;
  336) CONFIG="${SCRIPT_DIR}/qwen2_5vl_lora_sft_336.yaml"; DEFAULT_OUTPUT="saves/qwen2_5vl-7b/lora/uav_sft_336" ;;
  448) CONFIG="${SCRIPT_DIR}/qwen2_5vl_lora_sft_448.yaml"; DEFAULT_OUTPUT="saves/qwen2_5vl-7b/lora/uav_sft_448" ;;
  *) echo "Unknown profile: ${PROFILE}" >&2; exit 2 ;;
esac

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="run_${PROFILE}_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_DIR="${DEFAULT_OUTPUT}/${RUN_NAME}"
TMP_CONFIG="${LLAMAFACTORY_DIR}/.airnav_train_${PROFILE}_${RUN_NAME}.yaml"

cleanup() {
  rm -f "${TMP_CONFIG}"
}
trap cleanup EXIT

python "${SCRIPT_DIR}/prepare_llamafactory_dataset.py" \
  --generated_dir "${GENERATED_DIR}" \
  --llamafactory_dir "${LLAMAFACTORY_DIR}" \
  --assistant_format "${ASSISTANT_FORMAT}"

MATERIALIZE_ARGS=(
  --template "${CONFIG}"
  --output_dir "${OUTPUT_DIR}"
  --destination "${TMP_CONFIG}"
)
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  MATERIALIZE_ARGS+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
fi
python "${SCRIPT_DIR}/materialize_train_config.py" "${MATERIALIZE_ARGS[@]}"

cd "${LLAMAFACTORY_DIR}"
echo "[train] profile=${PROFILE} run_name=${RUN_NAME}"
echo "[train] output_dir=${OUTPUT_DIR}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "[train] resume_from_checkpoint=${RESUME_CHECKPOINT}"
fi
llamafactory-cli train "${TMP_CONFIG}"
