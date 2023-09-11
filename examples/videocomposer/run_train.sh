export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=6000
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 0 to simplify
export DEVICE_ID=7

output_path='outputs/train'

nohup python -u train.py \
    > $output_path/train.log 2>&1 &
