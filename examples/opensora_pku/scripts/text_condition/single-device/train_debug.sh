'''
Training scheduler
We replaced the eps-pred loss with v-pred loss and enable ZeroSNR. For videos, we resample to 16 FPS for training.

Stage 1: We initially initialized from the image weights of version 1.2.0 and trained images at a resolution of 1x320x320. The objective of this phase was to fine-tune the 3D dense attention model to a sparse attention model. The entire fine-tuning process involved approximately 100k steps, with a batch size of 1024 and a learning rate of 2e-5. The image data was primarily sourced from SAM in version 1.2.0.

Stage 2: We trained the model jointly on images and videos, with a maximum resolution of 93x320x320. 
The entire fine-tuning process involved approximately 300k steps, with a batch size of 1024 and a learning rate of 2e-5. 
The image data was primarily sourced from SAM in version 1.2.0, while the video data consisted of the unfiltered Panda70m. 
In fact, the model had nearly converged around 100k steps, and by 300k steps, there were no significant gains. 
Subsequently, we performed data cleaning and caption rewriting, with further data analysis discussed at the end.

Stage 3: We fine-tuned the model using our filtered Panda70m dataset, with a fixed resolution of 93x352x640. The entire fine-tuning process involved approximately 30k steps, with a batch size of 1024 and a learning rate of 1e-5.
'''

# Stage 2: 93x320x320
export DEVICE_ID=0
NUM_FRAME=29
HEIGHT=320
WIDTH=320
python  opensora/train/train_t2v_diffusers.py \
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
    --force_resolution \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --dataloader_num_workers 1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps 1000000 \
    --start_learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --seed=10 \
    --lr_warmup_steps=500 \
    --precision="bf16" \
    --checkpointing_steps=1000 \
    --output_dir="./checkpoints/t2v-${NUM_FRAME}x${HEIGHT}x${WIDTH}/" \
    --model_max_length 512 \
    --use_image_num 0 \
    --cfg 0.1 \
    --snr_gamma 5.0 \
    --use_ema True\
    --ema_start_step 0 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --use_rope \
    --noise_offset 0.02 \
    --enable_stable_fp32 True \
    --ema_decay 0.999 \
    --speed_factor 1.0 \
    --drop_short_ratio 1.0 \
    --hw_stride 32 \
    --sparse1d \
    --sparse_n 4 \
    --train_fps 16 \
    --trained_data_global_step 0 \
    --group_data \
    --prediction_type "v_prediction" \
    --mode 1
