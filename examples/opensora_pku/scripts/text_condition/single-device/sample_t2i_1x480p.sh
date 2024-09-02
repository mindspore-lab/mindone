export DEVICE_ID=0
python opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.2.0/1x480p \
    --num_frames 1 \
    --height 480 \
    --width 640 \
    --cache_dir "./" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae\
    --save_img_path "./sample_videos/prompt_list_0_1x480p" \
    --fps 24 \
    --guidance_scale 4.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
