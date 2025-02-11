export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

export MS_DATASET_SINK_QUEUE=4


# log level
export GLOG_v=2
num_frames=64

output_dir=outputs/stdit_dvm_cached_512x512x$num_frames

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train.py --config configs/opensora/train/stdit_512x512x64.yaml \
	--csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
	--vae_latent_folder datasets/sora_overfitting_dataset_0410_vae_512x512 \
	--text_embed_folder datasets/sora_overfitting_dataset_0410 \
    --use_parallel=True \
    --jit_level="O1" \
    --num_frames=$num_frames \
    --optim=adamw \
    --global_bf16=False \
    --dataset_sink_mode=True \
    --num_recompute_blocks=21 \
    --num_parallel_workers=16 \
    --enable_flash_attention=True \
    --gradient_accumulation_steps=1 \
    --use_ema=False \
    --output_path=$output_dir \

    # --max_device_memory="55GB" \
	# --video_folder datasets/sora_overfitting_dataset_0410 \
