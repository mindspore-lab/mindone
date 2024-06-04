export MS_DATASET_SINK_QUEUE=10

python train.py --config configs/training/mmv2_finetune_512.yaml \
    --data_path "/path/to/video_folder" \
    --csv_path "/path/to/video_caption.csv" \
    --video_column "videoid" \
    --caption_column "name" \
    --enable_flash_attention=False \
    --use_recompute=True \
    --output_path "outputs/mmv2_finetune_512" \
