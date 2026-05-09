#!/usr/bin/env bash
set -euo pipefail
SFT_DATA="$1"
TRANSFORMER_PATH="$2"
OUTPUT_DIR="$3"
OUT_PREFIX="$4"
PROMPT_TASK="$5"
Q_MAX_LEN="$6"
Q_TRUNC_SIDE="$7"
MODEL_TYPE="$8"
PROMPT_TYPE="$9"

mkdir -p "$OUTPUT_DIR"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python "${ROOT}/tools/preprocess_sft_data.py" \
  --input "${SFT_DATA}" \
  --tokenizer-name-or-path "${TRANSFORMER_PATH}" \
  --output-prefix "${OUTPUT_DIR}/${OUT_PREFIX}" \
  --task "${PROMPT_TASK}" \
  --tokenizer-type PretrainedFromHF \
  --dataset-impl mmap \
  --workers 4 \
  --question_max_length "${Q_MAX_LEN}" \
  --question_truncation_side "${Q_TRUNC_SIDE}" \
  --model_type "${MODEL_TYPE}" \
  --prompt_type "${PROMPT_TYPE}"
