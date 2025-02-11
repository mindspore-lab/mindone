export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

export MS_DATASET_SINK_QUEUE=4

# log level
export GLOG_v=2

# dynamic shape acceleration
export MS_DEV_ENABLE_KERNEL_PACKET=on

output_dir=outputs/opensora1.1_stage2_dynamic_shape

msrun --bind_core=True --master_port=8207 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train.py \
	--config configs/opensora-v1-1/train/train_stage2.yaml \
	--csv_path datasets/mixkit-100videos/video_caption_train.csv \
	--video_folder datasets/mixkit-100videos/mixkit \
	--text_embed_folder datasets/mixkit-100videos/t5_emb_200 \
  --use_parallel True \
  --jit_level O0 \
  --dataset_sink_mode=False \
  --num_parallel_workers=8 \
  --enable_flash_attention=True \
  --gradient_accumulation_steps=1 \
  --use_ema=False \
  --output_path=$output_dir
