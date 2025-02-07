export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

# log level
export GLOG_v=2

output_dir=output/stage1_t2i_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" \
python scripts/train.py \
  --config configs/train/stage1_t2i_256px.yaml \
  --env.mode 0 \
  --env.jit_level O1 \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --train.settings.zero_stage 2 \
  --dataset.csv_path CSV_PATH \
  --dataset.video_folder VIDEO_FOLDER \
  --dataset.text_emb_folder.ul2 UL2_FOLDER \
  --dataset.text_emb_folder.byt5 BYT5_FOLDER \
  --train.ema "" \
  --train.output_path "$output_dir"
