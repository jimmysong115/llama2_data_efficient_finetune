
# torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
#     --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
#     --data_path ./alpaca_data.json \
#     --bf16 True \
#     --output_dir <your_output_dir> \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --deepspeed "./configs/default_offload_opt_param.json" \
#     --tf32 True

#!/bin/bash -x
set -x

# Gemma model needs transformers==4.38.1

#llama3
pip install --upgrade transformers
pip install --upgrade accelerate

# pip install -U transformers==4.34.0
# pip install -U accelerate==0.27.2

pip install -U deepspeed==0.14.0
pip install -U openai==0.28.0

# torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) --master_port=20001 ${TRAIN_SCRIPT} \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_path ${DATA_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --bf16 ${BF16} \
#     --num_train_epochs ${NUM_TRAIN_EPOCHS} \
#     --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
#     --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
#     --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
#     --evaluation_strategy "${EVALUATION_STRATEGY}" \
#     --save_strategy "${SAVE_STRATEGY}" \
#     --save_steps ${SAVE_STEPS} \
#     --save_total_limit ${SAVE_TOTAL_LIMIT} \
#     --learning_rate ${LEARNING_RATE} \
#     --weight_decay ${WEIGHT_DECAY} \
#     --warmup_ratio ${WARMUP_RATIO} \
#     --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
#     --logging_steps ${LOGGING_STEPS} \
#     --deepspeed ${DEEPSPEED} \
#     --tf32 ${TF32} \
#     --model_max_length ${MODEL_MAX_LENGTH} \
#     --report_to ${REPORT_TO} \
#     --logging_dir "${LOGGING_DIR}"

# chmod -R 777 ${OUTPUT_MOUNT_PATH}
# chmod -R 777 ${OUTPUT_DIR}

torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) --master_port=20001 ${TRAIN_SCRIPT} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --bf16 ${BF16} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --evaluation_strategy "${EVALUATION_STRATEGY}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps ${LOGGING_STEPS} \
    --deepspeed ${DEEPSPEED} \
    --tf32 ${TF32} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --report_to ${REPORT_TO} \
    --logging_dir "${LOGGING_DIR}" \


chmod -R 777 ${OUTPUT_MOUNT_PATH}
chmod -R 777 ${OUTPUT_DIR}






