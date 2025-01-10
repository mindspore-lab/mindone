# Stage 1: 1x256x256
NUM_FRAME=1
WIDTH=256
HEIGHT=256
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="t2v-video-${NUM_FRAME}x${HEIGHT}x${WIDTH}/parallel_logs" \
  opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V_v1_3-2B/122 \
    --text_encoder_name_1 google/mt5-xxl \
    --cache_dir "./" \
    --dataset t2v \
    --data "scripts/train_data/image_data_v1_2.txt" \
    --ae WFVAEModel_D8_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --sample_rate 1 \
    --num_frames ${NUM_FRAME} \
    --force_resolution \
    --max_height ${HEIGHT} \
    --max_width ${WIDTH} \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --gradient_checkpointing \
    --train_batch_size=4 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --start_learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --seed=10 \
    --lr_warmup_steps=0 \
    --precision="bf16" \
    --checkpointing_steps=1000 \
    --output_dir="t2v-video-${NUM_FRAME}x${HEIGHT}x${WIDTH}/" \
    --model_max_length 512 \
    --use_image_num 0 \
    --cfg 0.1 \
    --snr_gamma 5.0 \
    --rescale_betas_zero_snr \
    --use_ema False \
    --ema_start_step 0 \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --noise_offset 0.02 \
    --ema_decay 0.999 \
    --speed_factor 1.0 \
    --drop_short_ratio 0.0 \
    --use_parallel True \
    --parallel_mode "zero" \
    --zero_stage 2 \
    --max_device_memory "59GB" \
    --dataset_sink_mode False \
    --prediction_type "v_prediction" \
    --hw_stride 32 \
    --sparse1d \
    --sparse_n 4 \
    --train_fps 16 \
    --trained_data_global_step 0 \
    --group_data \
    --pretrained path/to/last/stage/ckpt \
