# Quick debug for DiT config:
# - 1 NPU/GPU
# - fewer frames: 29
# - small and uncommon resolution: 352x640
# - fps: 24
# - precision: bf16 (Some exceptions ref to sample_utils.py.  Torch ver doesn't share, always uses fp16. )

# Debug first prompt only:
# "A young man at his 20s is sitting on a piece of cloud in the sky, reading a book."

# To use: 
# change model_path, text_encoder_name_1, ae_path, save_img_path before running the script.

export DEVICE_ID=0
python opensora/sample/sample.py \
    --model_path /home_host/susan/workspace/checkpoints/LanguageBind/Open-Sora-Plan-v1.3.0/any93x640x640 \
    --version v1_3 \
    --num_frames 29 \
    --height 352 \
    --width 640 \
    --text_encoder_name_1 /home_host/susan/workspace/checkpoints/google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae WFVAEModel_D8_4x8x8  \
    --ae_path /home_host/susan/workspace/checkpoints/LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --save_img_path "./sample_videos/prompt_list_0_29x640_mt5_bf16_debug" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --rescale_betas_zero_snr \
    --prediction_type "v_prediction" \
    --mode 1 --precision bf16 