# This script is for futher function
# Does not work yet for 2nd text encoder

export DEVICE_ID=0
python opensora/sample/sample.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.3.0/any93x640x640 \
    --version v1_3 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --text_encoder_name_1 google/mt5-xxl \
    --text_encoder_name_2 laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
    --text_prompt examples/prompt_list_0.txt \
    --ae WFVAEModel_D8_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --save_img_path "./sample_videos/prompt_list_0_93x640_mt5_clipbigG" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --num_samples_per_prompt 1 \
    --rescale_betas_zero_snr \
    --prediction_type "v_prediction" \
