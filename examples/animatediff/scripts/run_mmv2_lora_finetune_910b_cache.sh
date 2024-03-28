export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python train.py --config configs/training/mmv2_lora.yaml  \
    --image_size 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --output_path outputs/mmv2_lora_finetune_512bs1ga1_mindrecord  \
    --dataset_sink_mode True \
    --use_recompute False \
    --data_path ../videocomposer/datasets/webvid5_cache_mindrecord \
    --enable_flash_attention False \
    --sink_size 30 \
    --train_data_type mindrecord
