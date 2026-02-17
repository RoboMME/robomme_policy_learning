MODEL_TYPE="pi05_baseline"

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_baseline \
--exp-name=${MODEL_TYPE} \
--batch-size=64 \
--num-workers=4 \
--fsdp-devices=4 \
--dataset-path=data/robomme_preprocessed_data