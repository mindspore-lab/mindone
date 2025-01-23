# The DiT model is trained arbitrarily on stride=32.
# So keep the resolution of the inference a multiple of 32. Frames needs to be 4n+1, e.g. 93, 77, 61, 45, 29, 1 (image).

export DEVICE_ID=0
python opensora/sample/sample.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.3.0/1x256x256 \
    --version v1_3 \
    --num_frames 1 \
    --height 256 \
    --width 256 \
    --text_encoder_name_1 google/mt5-xxl \
    --text_prompt examples/prompt_list_human_images.txt \
    --ae WFVAEModel_D8_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --save_img_path "./sample_images/human_images" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --rescale_betas_zero_snr \
    --prediction_type "v_prediction" \
    --precision bf16 \
    --ms_checkpoint ckpt/path \
