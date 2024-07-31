python examples/rec_video.py \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae/causalvae_d4_488.ckpt \
    --image_path /storage/dataset/image/anytext3m/ocr_data/Art/images/gt_5544.jpg \
    --rec_path rec.jpg \
    --device Ascend \
    --sample_rate 1 \
    --short_size 512 \
    --enable_tiling
