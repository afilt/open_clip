#!/bin/bash
# ---------------------------------------------------------------------------
# PathGen-1.6M CLIP fine-tuning: H0-mini (vision) + BiomedBERT (text)
#
# Uses open_clip_train.main as the training entry point — identical to the
# reference h14_224_32_finetune.sh.  All losses, data loading, DDP,
# checkpointing and LR scheduling are provided by the open_clip repo.
#
# Supports:
#   Single-node single-GPU   bash scripts/train_pathology_clip.sh
#   Single-node multi-GPU    NPROC=4 bash scripts/train_pathology_clip.sh
#   Multi-node (torchrun)    Set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT
#                            before calling, or let the SLURM script do it.
#
# Key environment variables (override before calling):
#   TRAIN_DATA    path to pathgen_manifest.csv (default: /path/to/...)
#   OUTPUT_DIR    training log / checkpoint root
#   TCGA_ROOT     TCGA-UT data directory (optional, for mid-train eval)
#   SCORPION_ROOT SCORPION data directory (optional, for mid-train eval)
#   NPROC         GPUs per node          (default: all available or 1)
#   NNODES        total number of nodes  (default: 1)
#   NODE_RANK     rank of this node      (default: 0)
#   MASTER_ADDR   rendezvous host        (default: localhost)
#   MASTER_PORT   rendezvous port        (default: 29500)
#
# Extra CLI flags forwarded to open_clip_train.main:
#   bash scripts/train_pathology_clip.sh --epochs 5 --batch-size 16
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(realpath "$(dirname "$0")/..")"
SRC_DIR="${REPO_DIR}/src"

# Ensure the repo src is on PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Paths (override via environment)
# ---------------------------------------------------------------------------
TRAIN_DATA="${TRAIN_DATA:-/path/to/pathgen_manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/logs/pathgen_clip}"
TCGA_ROOT="${TCGA_ROOT:-/Users/afiliot/Desktop/tcga_ut/data}"
SCORPION_ROOT="${SCORPION_ROOT:-/Users/afiliot/Desktop/scorpion}"

# ---------------------------------------------------------------------------
# Distributed settings
# ---------------------------------------------------------------------------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [ -z "${NPROC:-}" ]; then
    # Auto-detect: number of CUDA GPUs, fall back to 1 (CPU/MPS)
    NPROC=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
fi

# Extra CLI flags from the caller
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
torchrun \
    --nproc_per_node="${NPROC}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m open_clip_train.main \
    \
    --model              H0-mini-BiomedBERT \
    --train-data         "${TRAIN_DATA}" \
    --csv-img-key        image_path \
    --csv-caption-key    caption \
    --csv-separator      , \
    --dataset-type       csv \
    --logs               "${OUTPUT_DIR}" \
    --save-frequency     1 \
    --report-to          tensorboard \
    \
    --apply-lora \
    --lora-r             8 \
    --lora-alpha         16 \
    --lora-dropout       0.05 \
    --lora-vision-target-modules qkv \
    --lora-text-target-modules   query key value \
    \
    --epochs             10 \
    --batch-size         32 \
    --lr                 1e-4 \
    --wd                 1e-4 \
    --warmup             200 \
    --accum-freq         2 \
    --grad-checkpointing \
    --precision          amp \
    \
    --workers            8 \
    \
    --image-mean         0.707223 0.578729 0.703617 \
    --image-std          0.211883 0.230117 0.177517 \
    --aug-cfg            "scale=(0.4, 1.0)" \
    \
    --eval-tcga-root     "${TCGA_ROOT}" \
    --eval-scorpion-root "${SCORPION_ROOT}" \
    --eval-interval      1 \
    --eval-batch-size    64 \
    --eval-max-tiles     50 \
    \
    "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
