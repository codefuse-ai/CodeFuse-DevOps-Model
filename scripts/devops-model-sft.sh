set -v 

torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --node_rank=$RANK src/train_bash.py \
    --deepspeed path_to_deepspeed_config \
    --stage sft \
    --model_name_or_path path_to_model_path \
    --do_train \
    --report_to 'tensorboard' \
    --dataset your_corpus \
    --template chatml \
    --finetuning_type full \
    --output_dir path_to_save_model_path \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.20 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --learning_rate 5e-6 \
    --plot_loss \
    --max_source_length=2048 \
    --max_target_length=2048 \
    --dataloader_num_workers 8 \
    --val_size 0.01 \
    --bf16 \
    --overwrite_output_dir \
    --max_grad_norm 1.0
    