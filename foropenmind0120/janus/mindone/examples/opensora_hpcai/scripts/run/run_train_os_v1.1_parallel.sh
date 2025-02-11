export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

export MS_DATASET_SINK_QUEUE=4

# enable kbk: 1
#export MS_ENABLE_ACLNN=1
#export GRAPH_OP_RUN=1

# log level
export GLOG_v=2
num_frames=16

output_dir=outputs/stdit2_512x512x$num_frames

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train.py \
	--config configs/opensora-v1-1/train/train_stage1.yaml \
	--csv_path datasets/sora_overfitting_dataset_0410/vcg_200_with_length.csv \
	--video_folder datasets/sora_overfitting_dataset_0410 \
	--text_embed_folder datasets/sora_overfitting_dataset_0410/t5_emb_200/video200/ \
  --use_parallel True \
  --num_frames=$num_frames \
  --dataset_sink_mode=False \
  --num_parallel_workers=8 \
  --enable_flash_attention=True \
  --gradient_accumulation_steps=1 \
  --use_ema=False \
  --output_path=$output_dir
