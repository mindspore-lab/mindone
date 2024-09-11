
NUM_FRAME=29
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="t2v-video3d-${NUM_FRAME}x720p_ddp/parallel_logs" \
  opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "./" \
    --dataset t2v \
    --data "scripts/train_data/merge_data_panda70m.txt" \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "LanguageBind/Open-Sora-Plan-v1.2.0/vae" \
    --sample_rate 1 \
    --num_frames ${NUM_FRAME} \
    --max_height 720 \
    --max_width 1280 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.5 \
    --interpolation_scale_w 2.0 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --start_learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --seed=10 \
    --lr_warmup_steps=500 \
    --precision="bf16" \
    --checkpointing_steps=1000 \
    --output_dir="t2v-video3d-${NUM_FRAME}x720p_ddp/" \
    --model_max_length 512 \
    --use_image_num 0 \
    --cfg 0.1 \
    --snr_gamma 5.0 \
    --use_ema True\
    --ema_start_step 0 \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --use_rope \
    --noise_offset 0.02 \
    --enable_stable_fp32 True\
    --ema_decay 0.999 \
    --speed_factor 1.0 \
    --drop_short_ratio 1.0 \
    --pretrained "LanguageBind/Open-Sora-Plan-v1.2.0/29x480p" \
    --use_parallel True \
    # --group_frame \
