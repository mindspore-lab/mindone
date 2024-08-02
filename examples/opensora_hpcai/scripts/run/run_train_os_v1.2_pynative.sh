export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

# log level
export GLOG_v=2

output_dir=outputs/OSv1.2_pynative

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train.py \
	--mode=1 \
	--config configs/opensora-v1-2/train/train_720x1280x51.yaml \
    --csv_path datasets/mixkit-100videos/video_caption_train.csv \
    --video_folder datasets/mixkit-100videos/mixkit \
    --text_embed_folder  datasets/mixkit-100videos/t5_emb_300 \
  --use_parallel True \
  --dataset_sink_mode=False \
  --enable_flash_attention=True \
  --gradient_accumulation_steps=1 \
  --use_ema=True \
  --output_path=$output_dir \
  --use_recompute=True \
  --vae_dtype=fp16  # FIXME: switch to bf16 when AMP issue is fixed
