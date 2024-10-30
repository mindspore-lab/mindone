export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0
export MS_DATASET_SINK_QUEUE=4

# log level
export GLOG_v=2

output_dir=output/moviegen_t2i_256x256/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --log_dir="$output_dir"  \
python train.py \
  --config configs/train/moviegen_t2i_256x256.yaml \
  --env.mode 0 \
  --env.jit_level O0 \
  --env.max_device_memory 59GB \
  --env.distributed=True \
  --model.name llama-1B \
  --dataset.csv_path CSV_PATH \
  --dataset.video_folder VIDEO_FOLDER \
  --dataset.text_emb_folder.ul2 UL2_FOLDER \
  --dataset.text_emb_folder.byt5 BYT5_FOLDER \
  --train.output_path=$output_dir \
  --train.ema ""  # turn off ema
