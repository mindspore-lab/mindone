# Stage 2: 93x320x320
NUM_FRAME=29
WIDTH=320
HEIGHT=320
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
msrun --bind_core=True --worker_num=4 --local_worker_num=4 --master_port=6000 --log_dir="./checkpoints/t2v-video-${NUM_FRAME}x${HEIGHT}x${WIDTH}_zero2_mode1_npu4/parallel_logs" \
  opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V_v1_3-2B/122 \
    --text_encoder_name_1 /home_host/susan/workspace/checkpoints/google/mt5-xxl \
    --cache_dir "./" \
    --dataset t2v \
    --data "scripts/train_data/video_data_v1_2.txt" \
    --ae WFVAEModel_D8_4x8x8  \
    --ae_path /home_host/susan/workspace/checkpoints/LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --sample_rate 1 \
    --num_frames ${NUM_FRAME} \
    --max_height ${HEIGHT} \
    --max_width ${WIDTH} \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --start_learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --seed=10 \
    --lr_warmup_steps=500 \
    --precision="bf16" \
    --checkpointing_steps=1000 \
    --output_dir="./checkpoints/t2v-video-${NUM_FRAME}x${HEIGHT}x${WIDTH}_zero2_mode1_npu4/" \
    --model_max_length 512 \
    --use_image_num 0 \
    --cfg 0.1 \
    --snr_gamma 5.0 \
    --use_ema False \
    --ema_start_step 0 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --noise_offset 0.02 \
    --enable_stable_fp32 True\
    --ema_decay 0.999 \
    --speed_factor 1.0 \
    --drop_short_ratio 1.0 \
    --use_parallel True \
    --parallel_mode "zero" \
    --zero_stage 2 \
    --max_device_memory "58GB" \
    --jit_syntax_level "lax" \
    --dataset_sink_mode True \
    --num_no_recompute 18 \
    --prediction_type "v_prediction" \
    --hw_stride 32 \
    --sparse1d \
    --sparse_n 4 \
    --train_fps 16 \
    --trained_data_global_step 0 \
    --group_data \
    --mode 1
