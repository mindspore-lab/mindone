# export ASCEND_RT_VISIBLE_DEVICES=7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0
export MS_DATASET_SINK_QUEUE=8

# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

output_dir=outputs/debug_train_tae_1p_sd3.5vaeInit_noOpl

python scripts/train_tae.py \
--mode=0 \
--jit_level O0 \
--amp_level O0 \
--use_outlier_penalty_loss False \
--dtype fp32 \
--config configs/tae/train/video_ft.yaml \
--output_path=$output_dir \
--epochs=2000 --ckpt_save_interval=50 \

# --use_parallel=True \
