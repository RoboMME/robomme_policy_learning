MODEL_TYPE="pi05_baseline"

export WANDB_API_KEY=a8ec9b33c5c4ccaf628b79412e66dbaac2f7009d

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_baseline_fullfinetune \
--exp-name=pi05_baseline_fullfinetune_robomme \
--batch-size=64 \
--num-workers=8 \
--fsdp-devices=1 \
--dataset-path=data/robomme_preprocessed_data \
--overwrite