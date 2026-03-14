#!/bin/bash
#SBATCH --job-name=pathgen_clip
#SBATCH --output=logs/pathgen_clip_%j.out
#SBATCH --error=logs/pathgen_clip_%j.err
#SBATCH --nodes=1                    # increase for multi-node
#SBATCH --ntasks-per-node=4          # 4 GPUs per node
#SBATCH --cpus-per-task=8            # 8 CPU cores per GPU
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
# Uncomment and adjust for your cluster:
# #SBATCH --account=<your_account>
# #SBATCH --constraint=a100

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
set -euo pipefail

# Activate your conda / venv environment:
# source activate pathology_clip
# source /path/to/venv/bin/activate

module purge
# module load cuda/12.1
# module load cudnn/8.9

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# Paths (edit these)
# ---------------------------------------------------------------------------
REPO_DIR="$(realpath "$(dirname "$0")/..")"

export TRAIN_DATA="/path/to/pathgen_manifest.csv"
export OUTPUT_DIR="${REPO_DIR}/logs/pathgen_clip_slurm"
export TCGA_ROOT="/Users/afiliot/Desktop/tcga_ut/data"
export SCORPION_ROOT="/Users/afiliot/Desktop/scorpion"

# ---------------------------------------------------------------------------
# Multi-node distributed settings (wired from SLURM automatically)
# torchrun inside train_pathology_clip.sh reads these env vars.
# ---------------------------------------------------------------------------
export NPROC="${SLURM_NTASKS_PER_NODE}"
export NNODES="${SLURM_NNODES}"
export NODE_RANK="${SLURM_NODEID}"
# Use the first node in the allocation as the rendezvous host
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p logs "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Launch — one call per node (SLURM launches this script on every node)
# train_pathology_clip.sh calls torchrun, which handles intra-node DDP.
# Cross-node communication uses the c10d rendezvous above.
# ---------------------------------------------------------------------------
bash "${REPO_DIR}/scripts/train_pathology_clip.sh" \
    --epochs        10 \
    --batch-size    32 \
    --accum-freq    2

echo "Node ${SLURM_NODEID} / job ${SLURM_JOB_ID} done."
