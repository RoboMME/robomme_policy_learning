#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=a8ec9b33c5c4ccaf628b79412e66dbaac2f7009d

CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
uv run scripts/train_dense_history.py pi05_baseline_with_history \
  --exp-name=pi05_dense_history \
  --batch-size=16 \
  --num-workers=4 \
  --fsdp-devices=1 \
  --dataset-path=data/robomme_preprocessed_data \
  --overwrite
