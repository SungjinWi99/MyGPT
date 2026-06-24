#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/content/drive/MyDrive/KTB/MyGPT/datasets}"
OUTPUT_VERSION="${OUTPUT_VERSION:-v1}"
WORK_DIR="${WORK_DIR:-/content/mygpt_dataset_work}"
RAW_CACHE_DIR="${RAW_CACHE_DIR:-${DATASET_ROOT}/raw}"
AIHUB_ROOT="${AIHUB_ROOT:-/content/drive/MyDrive/KTB/datasets/AIHUB}"

sources=(wikimedia webtext)
if [[ "${INCLUDE_AIHUB:-0}" == "1" ]]; then
  sources+=(aihub)
fi

args=(
  python -m src.dataset_pipeline.prepare_pretrain
  --output-dir "${DATASET_ROOT}/pretrain/${OUTPUT_VERSION}"
  --work-dir "${WORK_DIR}"
  --raw-cache-dir "${RAW_CACHE_DIR}"
  --sources "${sources[@]}"
  --webtext-upstream-sources news science-webtext law
  --shard-rows 100000
  --tokenize-batch-size 128
  --validation-fraction 0.005
)

if [[ "${INCLUDE_AIHUB:-0}" == "1" ]]; then
  args+=(--aihub-root "${AIHUB_ROOT}")
fi

if [[ -n "${MAX_ACCEPTED_PER_SOURCE:-}" ]]; then
  args+=(--max-accepted-per-source "${MAX_ACCEPTED_PER_SOURCE}")
fi

if [[ -n "${WEBTEXT_ALLOW_PATTERN:-}" ]]; then
  args+=(--webtext-allow-pattern "${WEBTEXT_ALLOW_PATTERN}")
fi

"${args[@]}"
