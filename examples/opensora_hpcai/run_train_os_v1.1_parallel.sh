export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

export MS_DATASET_SINK_QUEUE=4

# enable kbk: 1
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

# log level
export GLOG_v=2
num_frames=64

output_dir=outputs/stdit2_ms2.3rc2_512x512x$num_frames

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train.py \
	--config configs/opensora-v1-1/train/stdit2_512x512x64.yaml \
	--csv_path datasets/sora_overfitting_dataset_0410/vcg_200_with_length.csv \
	--text_embed_folder datasets/sora_overfitting_dataset_0410/t5_emb_200 \
    --video_folder datasets/sora_overfitting_dataset_0410 \
    --vae_latent_folder datasets/sora_overfitting_dataset_0410_vae_512x512 \
  --use_parallel True \
  --num_frames=$num_frames \
  --dataset_sink_mode=False \
  --num_parallel_workers=8 \
  --enable_flash_attention=True \
  --output_path=$output_dir \

