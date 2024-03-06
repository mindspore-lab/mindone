export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python train.py --config configs/training/mmv2_train.yaml \
    --data_path "../videocomposer/datasets/webvid5" \
    --csv_path "../videocomposer/datasets/webvid5_copy.csv" \
    --output_path "outputs/mmv2_train_webvid5" \
    --enable_flash_attention=False \
    --use_recompute=True \
    --dataset_sink_mode=True \
    --sink_size 100 \
    --train_steps=16000 \
    --ckpt_save_steps=4000 \
    --train_batch_size 1 \
    --log_interval 1 \
    --image_size 512 \
