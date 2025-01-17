export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

# operation/graph fusion for dynamic shape
# export MS_DEV_ENABLE_KERNEL_PACKET=on # TODO: add dynamic shape support

# log level
export GLOG_v=2

output_dir=output/stage2_t2iv_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" \
python scripts/train.py \
  --config configs/train/stage2_t2iv_256px.yaml \
  --env.mode 0 \
  --env.jit_level O1 \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.name=llama-30B \
  --train.settings.zero_stage 3 \
  --train.sequence_parallel.shards 8 \
  --dataset.csv_path CSV_PATH \
  --dataset.video_folder VIDEO_FOLDER \
  --dataset.tae_latent_folder TAE_LATENT_FOLDER \
  --dataset.text_emb_folder.ul2 UL2_FOLDER \
  --dataset.text_emb_folder.byt5 BYT5_FOLDER \
  --dataset.sample_n_frames 32 \
  --dataloader.batch_size 1 \
  --train.ema "" \
  --train.output_path "$output_dir"
