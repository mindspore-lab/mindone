export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python train.py --config configs/training/mmv2_lora.yaml \
    --data_path "/mnt/disk3/datasets/webvid/2M_train/part5/00000"  \
    --output_path "outputs/mmv2_lora_webvid10k_PR" \
    --dataset_sink_mode=True \
    --sink_size=100 \
    --train_steps=1000 \
    --ckpt_save_steps=1000 \
    --train_batch_size=1 \
    --log_interval=1 \
    --use_recompute=True \
    --enable_flash_attention=False \
    --image_size 512 \
