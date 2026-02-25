#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <num-examples>"
  exit 1
fi

NUM_EXAMPLES="$1"
if ! [[ "$NUM_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "num-examples must be a positive integer"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".env" ]]; then
  echo "Missing .env at $SCRIPT_DIR/.env"
  exit 1
fi

set -a
source .env
set +a

BASE_DATASET_PREFIX="${HF_DATASET_PREFIX:-collinear-ai/zoho-1f-tools}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"

STAGE1_DATASET="${BASE_DATASET_PREFIX}-prompts-for-model-${RUN_ID}"
STAGE2_DATASET="${BASE_DATASET_PREFIX}-prompts-${RUN_ID}"
STAGE3_DATASET="${BASE_DATASET_PREFIX}-response-${RUN_ID}"

echo "Stage 1 dataset: ${STAGE1_DATASET}"
echo "Stage 2 dataset: ${STAGE2_DATASET}"
echo "Stage 3 dataset: ${STAGE3_DATASET}"

python generate_model_prompts.py \
  --num-examples "${NUM_EXAMPLES}" \
  --dataset-id "${STAGE1_DATASET}"

python generate_prompts.py \
  --dataset-in "${STAGE1_DATASET}" \
  --dataset-out "${STAGE2_DATASET}"

python generate_tool_calls.py \
  --dataset-in "${STAGE2_DATASET}" \
  --dataset-out "${STAGE3_DATASET}"

echo "Done."
echo "stage1=${STAGE1_DATASET}"
echo "stage2=${STAGE2_DATASET}"
echo "stage3=${STAGE3_DATASET}"
