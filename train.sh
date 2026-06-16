#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/content/drive/MyDrive/KTB/MyGPT/datasets}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/content/drive/MyDrive/KTB/MyGPT/checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-./config.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-MyGPT}"

python -m src.train \
  --dataset-path "${DATASET_ROOT}/pretrain/${DATASET_VERSION}" \
  --weights-dir "${WEIGHTS_DIR}" \
  --config-path "${CONFIG_PATH}" \
  --wandb "${WANDB_PROJECT}"
