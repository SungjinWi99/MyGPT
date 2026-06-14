#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/content/drive/MyDrive/KTB/MyGPT/datasets}"
OUTPUT_VERSION="${OUTPUT_VERSION:-v1}"
WORK_DIR="${WORK_DIR:-/content/mygpt_dataset_work}"
RAW_CACHE_DIR="${RAW_CACHE_DIR:-${DATASET_ROOT}/raw}"

args=(
  python -m src.dataset_pipeline.prepare_pretrain
  --output-dir "${DATASET_ROOT}/pretrain/${OUTPUT_VERSION}"
  --work-dir "${WORK_DIR}"
  --raw-cache-dir "${RAW_CACHE_DIR}"
  --sources wikimedia historical
  --shard-rows 100000
  --tokenize-batch-size 128
  --validation-fraction 0.005
)

if [[ -n "${MAX_ACCEPTED_PER_SOURCE:-}" ]]; then
  args+=(--max-accepted-per-source "${MAX_ACCEPTED_PER_SOURCE}")
fi

if [[ -n "${HISTORICAL_ALLOW_PATTERN:-}" ]]; then
  args+=(--historical-allow-pattern "${HISTORICAL_ALLOW_PATTERN}")
fi

"${args[@]}"
