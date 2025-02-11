export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python train.py --config configs/training/mmv2_train.yaml \
    --data_path "/mnt/disk3/datasets/webvid/2M_train/part5/00000"  \
    --output_path "outputs/mmv2_train_webvid10k" \
    --enable_flash_attention=False \
    --use_recompute=True \
    --dataset_sink_mode=True \
    --sink_size 100 \
    --train_steps=10000 \
    --ckpt_save_steps=1000 \
    --train_batch_size 1 \
    --log_interval 1 \
    --image_size 512 \
