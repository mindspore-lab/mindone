export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

# <1.2s
python train.py --config configs/training/mmv2_lora.yaml \
    --data_path "/mnt/disk3/datasets/webvid/2M_train/part5/00000"  \
    --output_path "outputs/mmv2_lora_webvid10k" \
    --dataset_sink_mode True\
    --sink_size 100 \
    --epochs  100 \
    --ckpt_save_interval 100 \
    --log_interval 1 \
    --use_recompute True \
    --recompute_strategy 'down_mm' \
    --enable_flash_attention False \
    --image_size 512 \
    --train_batch_size 1 \

:' Uncommet 
# slower
python train.py --config configs/training/mmv2_lora.yaml \
    --data_path "/mnt/disk3/datasets/webvid/2M_train/part5/00000"  \
    --output_path "outputs/mmv2_lora_webvid10k_fa" \
    --dataset_sink_mode True\
    --sink_size 100 \
    --epochs  100 \
    --ckpt_save_interval 100 \
    --log_interval 1 \
    --use_recompute False \
    --enable_flash_attention True \
    --image_size 512 \
    --train_batch_size 1 \
'
