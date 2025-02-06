# Stage 4: 29x720p two-nodes, 16 NPUs
NUM_FRAME=29

MS_WORKER_NUM=16                      # the total number of workers in all nodes
LOCAL_WORKER_NUM=8                    # the number of workers in the current node
NODE_RANK=$1                          # the ID of the current node, pass it via `bash xxx.sh 0` or `bash xxx.sh 1`
MASTER_NODE_ADDRESS="x.xxx.xxx.xxx"   # the address of the master node. Use the same master address in two nodes
echo "Running on node rank $NODE_RANK"
msrun --bind_core=True --node_rank=$NODE_RANK --worker_num=$MS_WORKER_NUM --local_worker_num=$LOCAL_WORKER_NUM --master_addr=$MASTER_NODE_ADDRESS --log_dir="node-${NODE_RANK}-t2v-video3d-${NUM_FRAME}x720p_zero2_sp/parallel_logs" \
  opensora/train/train_t2v_diffusers.py \
    --model OpenSoraT2V-ROPE-L/122 \
    --text_encoder_name google/mt5-xxl \
    --cache_dir "./" \
    --dataset t2v \
    --data "scripts/train_data/merge_data_mixkit.txt" \
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
    --output_dir="node-${NODE_RANK}-t2v-video3d-${NUM_FRAME}x720p_zero2_sp/" \
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
    --parallel_mode "zero" \
    --zero_stage 2 \
    --sp_size 8 \
    --train_sp_batch_size 1 \
    --max_device_memory "59GB" \
    --jit_syntax_level "lax" \
    --dataset_sink_mode False \
    # --gradient_checkpointing \
    # --group_frame \
