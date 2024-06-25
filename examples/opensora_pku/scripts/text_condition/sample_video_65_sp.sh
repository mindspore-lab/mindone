export ASCEND_RT_VISIBLE_DEVICES=0,1

output_dir=sample_log

msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9001 --log_dir=$output_dir/parallel_logs opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.1.0 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_65.txt \
    --ae CausalVAEModel_4x8x8 \
    --version 65x512x512 \
    --num_frames 65 \
    --height 512 \
    --width 512 \
    --save_img_path "./sample_videos_sp/prompt_list_65" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 150 \
    --enable_flash_attention "True" \
    --enable_tiling \
    --use_parallel True \
    --sp_size 2
