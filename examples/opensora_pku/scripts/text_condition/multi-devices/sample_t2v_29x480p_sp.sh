
export ASCEND_RT_VISIBLE_DEVICES=0,1
msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9000 --log_dir="./sample_videos/prompt_list_0_29x480p_sp/parallel_logs/" \
 opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.2.0/29x480p \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "./" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae\
    --save_img_path "./sample_videos/prompt_list_0_29x480p_sp" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
    --use_parallel True \
    --sp_size 2
