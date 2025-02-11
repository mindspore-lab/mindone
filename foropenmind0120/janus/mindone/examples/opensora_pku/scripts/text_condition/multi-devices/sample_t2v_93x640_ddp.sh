export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="./sample_videos/sora_93x640_mt5_ddp/parallel_logs/" \
   opensora/sample/sample.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.3.0/any93x640x640 \
    --version v1_3 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --text_encoder_name_1 google/mt5-xxl \
    --text_prompt examples/sora.txt \
    --ae WFVAEModel_D8_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --save_img_path "./sample_videos/sora_93x640_mt5_ddp" \
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
    --use_parallel True \
