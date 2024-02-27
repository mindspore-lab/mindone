export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python train.py --config configs/training/mmv2_train_cache.yaml \
    --image_size 512 \
    --train_batch_size 1 \
    --output_path outputs/mmv2_train_webvid5_512b1_cache_mindrecord \
    --data_path ../videocomposer/datasets/webvid5_cache_mindrecord \
    --use_recompute True \
    --dataset_sink_mode True \
    --train_data_type mindrecord
