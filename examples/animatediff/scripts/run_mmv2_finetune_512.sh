export MS_DATASET_SINK_QUEUE=10
#    --data_path "/path/to/video_folder" \
#    --csv_path "/path/to/video_caption.csv" \

python train.py --config configs/training/mmv2_finetune_512.yaml \
    --data_path "/home/hyx/datasets/wx_dataset" \
    --csv_path "/home/hyx/datasets/wx_dataset/train_data_0311_vision_china.csv" \
    --video_column "videoid" \
    --caption_column "name" \
    --enable_flash_attention=False \
    --use_recompute=True \
    --output_path "outputs/mmv2_finetune_512" \
