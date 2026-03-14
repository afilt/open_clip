#!/bin/bash
# ---------------------------------------------------------------------------
# Quick local training run on Mac (MPS / CPU, no real tiles needed).
#
# Drives open_clip_train.main directly — identical infrastructure to the
# cluster run, just with a tiny synthetic dataset and minimal LoRA.
#
# Flags used:
#   --dataset-type synthetic  → open_clip's built-in random image-text pairs
#   --apply-lora              → freeze base weights, train only LoRA + heads
#   --lora-r 2 / --lora-alpha 4  → tiny adapters to fit in Mac RAM
#   --precision fp32          → MPS does not support AMP
#   --workers 0               → avoid multiprocessing fork issues on macOS
#
# Usage:
#   bash scripts/run_local.sh
#   bash scripts/run_local.sh --epochs 1   # even faster smoke-test
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(realpath "$(dirname "$0")/..")"
SRC_DIR="${REPO_DIR}/src"

# Ensure the repo src is on PYTHONPATH so open_clip is importable
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

EXTRA_ARGS=("$@")

python -m open_clip_train.main \
    --model              H0-mini-BiomedBERT \
    --dataset-type       synthetic \
    --train-num-samples  512 \
    --logs               "${REPO_DIR}/logs/local_test" \
    --save-frequency     1 \
    \
    --apply-lora \
    --lora-r             2 \
    --lora-alpha         4 \
    --lora-dropout       0.05 \
    --lora-vision-target-modules qkv \
    --lora-text-target-modules   query key value \
    \
    --epochs             3 \
    --batch-size         4 \
    --lr                 1e-4 \
    --wd                 1e-4 \
    --warmup             10 \
    --accum-freq         1 \
    --precision          fp32 \
    --workers            0 \
    \
    --image-mean         0.707223 0.578729 0.703617 \
    --image-std          0.211883 0.230117 0.177517 \
    \
    "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
