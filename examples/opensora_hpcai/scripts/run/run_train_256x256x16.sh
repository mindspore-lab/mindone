# uncomment it to prepare the t5 text embedding data
# python scripts/infer_t5.py --csv_path "../videocomposer/datasets/webvid5/video_caption.csv" --output_path "../videocomposer/datasets/webvid5_t5" \

python scripts/train.py --config configs/opensora/train/stdit_256x256x16_ms.yaml \
    --csv_path "../videocomposer/datasets/webvid5/video_caption.csv" \
    --video_folder "../videocomposer/datasets/webvid5" \
    --text_embed_folder "../videocomposer/datasets/webvid5_t5" \
