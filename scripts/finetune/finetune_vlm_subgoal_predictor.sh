# More usage of Swift, please refer to https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html


# data/vlm_subgoal_prediction_data/qwenvl/simple_subgoal_train.jsonl
# data/vlm_subgoal_prediction_data/memer/grounded_subgoal_train.jsonl
DATASET_PATH='data/vlm_subgoal_prediction_data/qwenvl/simple_subgoal_train.jsonl'
OUTPUT_DIR='runs/ckpts/vlm_subgoal_predictor/qwenvl_simple_subgoal'

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=256 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS_MAX_FRAMES=10 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model 'Qwen/Qwen3-VL-4B-Instruct' \
    --dataset $DATASET_PATH \
    --split_dataset_ratio 0.0 \
    --load_from_cache_file true \
    --packing false \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 100 \
    --max_length 3200 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4