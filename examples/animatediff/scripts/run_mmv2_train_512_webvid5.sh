# export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
# export MS_DATASET_SINK_QUEUE=10

# If resume training:
# --resume "outputs/mmv2_train_webvid5_ms2.3PoC_fa/2024-03-09T14-29-02/ckpt/train_resume.ckpt" \

# If reduce epoch switch cost
#    --data_path "../videocomposer/datasets/webvid5" \
#    --csv_path "../videocomposer/datasets/webvid5_copy.csv" \

python train.py --config configs/training/mmv2_train.yaml \
    --data_path "../videocomposer/datasets/webvid5" \
    --csv_path "../videocomposer/datasets/webvid5_copy.csv" \
    --output_path "outputs/mmv2_train_webvid5_vFlip" \
    --enable_flash_attention=False \
    --use_recompute=True \
    --recompute_strategy="down_mm_half" \
    --dataset_sink_mode=True \
    --sink_size=100 \
    --train_steps=40000 \
    --ckpt_save_steps=4000 \
    --train_batch_size 1 \
    --image_size 512 \
    --start_learning_rate=1e-4 \
    --random_drop_text_ratio=0.1 \
    --disable_flip=False \
