#!/usr/bin/env bash
#SBATCH --job-name=mme-vla-ood-framesamp-modul
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=48G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/daiyp/MME-VLA-Suite/runs/logs/slurm/%x-%j.log

# Submit from the repository root after creating the log directory:
#   mkdir -p runs/logs/slurm
#   sbatch scripts/slurm/finetune_robomme_ood_framesamp_modul.sh
# Validate paths and the parsed configuration without allocating GPUs:
#   bash scripts/slurm/finetune_robomme_ood_framesamp_modul.sh --dry-run
# On another machine, set MME_VLA_REPO_DIR and MME_VLA_PYTHON as needed;
# override the Slurm log path with sbatch --output=/existing/directory/%x-%j.log.

set -euo pipefail

dry_run=0
if [[ ${1:-} == --dry-run && $# == 1 ]]; then
    dry_run=1
elif (( $# != 0 )); then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

REPO_DIR="${MME_VLA_REPO_DIR:-/home/daiyp/MME-VLA-Suite}"
TRAIN_PYTHON="${MME_VLA_PYTHON:-$REPO_DIR/.venv/bin/python}"
SOURCE_CHECKPOINT="${MME_VLA_SOURCE_CHECKPOINT:-$REPO_DIR/runs/ckpts/mme_vla_suite_v0/perceptual-framesamp-modul/79999}"
DATASET_PATH="${MME_VLA_DATASET_PATH:-/nfs/turbo/coe-chaijy-unreplicated/RoboMME/robomme_ood_preprocessed_data_1021}"
CHECKPOINT_BASE_DIR="${MME_VLA_CHECKPOINT_BASE_DIR:-$REPO_DIR/runs/ckpts}"
EXP_NAME="${MME_VLA_EXP_NAME:-perceptual-framesamp-modul-ood-1021-10k}"

cd "$REPO_DIR"
[[ -x "$TRAIN_PYTHON" ]] || { printf 'Missing Python environment: %s\n' "$TRAIN_PYTHON" >&2; exit 1; }
[[ -d "$SOURCE_CHECKPOINT/params" ]] || { printf 'Missing source checkpoint: %s\n' "$SOURCE_CHECKPOINT" >&2; exit 1; }
[[ -f "$SOURCE_CHECKPOINT/assets/robomme/norm_stats.json" ]] || { printf 'Missing source normalization statistics\n' >&2; exit 1; }
[[ -f "$DATASET_PATH/meta/stats.json" ]] || { printf 'Missing preprocessed dataset metadata: %s\n' "$DATASET_PATH" >&2; exit 1; }

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/VLA/openpi}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# The new OOD run starts a fresh optimizer and LR schedule from the trained
# model weights. Do not use --resume or --overwrite for this new experiment.
# keep-period retains each 1k checkpoint; the final folder is 9999 (zero-based).
train_args=(
    mme_vla_suite
    --exp-name="$EXP_NAME"
    --dataset-path="$DATASET_PATH"
    --model.use-history
    --model.history-config=perceptual-framesamp-modul.yaml
    --weight-loader.params-path="$SOURCE_CHECKPOINT/params"
    --data.assets.assets-dir="$SOURCE_CHECKPOINT/assets"
    --data.assets.asset-id=robomme
    --checkpoint-base-dir="$CHECKPOINT_BASE_DIR"
    --batch-size=64
    --num-workers=4
    --fsdp-devices=4
    --num-train-steps=10000
    --lr-schedule.warmup-steps=1000
    --lr-schedule.peak-lr=1e-5
    --lr-schedule.decay-steps=10000
    --lr-schedule.decay-lr=1e-6
    --save-interval=1000
    --keep-period=1000
    --log-interval=100
    --no-wandb-enabled
)

if (( dry_run )); then
    MME_VLA_VALIDATE_ONLY=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        "$TRAIN_PYTHON" -m scripts.train_ood "${train_args[@]}"
else
    : "${SLURM_JOB_ID:?Submit this script with sbatch, or use --dry-run for validation}"
    # Preserve CUDA_VISIBLE_DEVICES assigned by Slurm to this four-GPU job.
    unset MME_VLA_VALIDATE_ONLY
    srun --ntasks=1 "$TRAIN_PYTHON" -m scripts.train_ood "${train_args[@]}"
fi
