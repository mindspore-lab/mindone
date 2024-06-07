python opensora/train/train_causalvae.py \
    --exp_name "exp_name" \
    --batch_size 1 \
    --precision fp16 \
    --max_steps 40000 \
    --save_steps 100 \
    --output_dir results/causalvae \
    --video_path /remote-home1/dataset/data_split_tt \
    --video_num_frames 17 \
    --resolution 256 \
    --sample_rate 1 \
    --num_parallel_workers 8 \
    --load_from_checkpoint pretrained/causal_vae_488_init.ckpt \
    # --data_file_path path/to/sharegpt4v_path_cap_64x512x512.json \
    # --video_column "path" \
