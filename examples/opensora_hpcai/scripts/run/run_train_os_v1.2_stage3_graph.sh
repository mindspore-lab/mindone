export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0
export MS_DATASET_SINK_QUEUE=4

# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

output_dir=outputs/OSv1.2_graph_stage3/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train.py \
	--mode=0 \
	--jit_level=O1 \
	--max_device_memory 59GB \
	--config configs/opensora-v1-2/train/train_stage3.yaml \
	--csv_path YOUR_CSV_PATH \
	--video_folder YOUR_VIDEO_FOLDER \
	--text_embed_folder YOUR_TEXT_EMBED_FOLDER \
  --use_parallel True \
  --dataset_sink_mode=False \
  --enable_flash_attention=True \
  --gradient_accumulation_steps=1 \
  --use_ema=True \
  --output_path=$output_dir \
  --add_datetime=False \
  --vae_dtype=fp16
