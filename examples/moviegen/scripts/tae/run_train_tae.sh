# improve data loading performance for distributed training: 1
# export MS_ENABLE_NUMA=1

# plot memory usage and compile info
# export MS_DEV_RUNTIME_CONF="memory_statistics:True,compile_statistics:True"

# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

output_dir=outputs/train_tae

python scripts/train_tae.py \
--config configs/tae/train/mixed_256x256x32.yaml \
--use_outlier_penalty_loss False \
--csv_path datasets/ucf101_train.csv \
--folder datasets/UCF-101 \
--output_path=$output_dir \
--epochs=100 --ckpt_save_interval=5 \
