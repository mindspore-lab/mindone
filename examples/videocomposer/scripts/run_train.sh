export MS_ENABLE_REF_MODE=0 # will be set to 1 in latest ms version. TODO: remove for future MS version
export MS_ASCEND_CHECK_OVERFLOW_MODE=1 #"INFNAN_MODE" # for ms+910B, check overflow
#export INF_NAN_MODE_ENABLE=1 # For pytorch+npu, recommend to enable it for mixed precision training for 910B. it determines how overflow is detected

export GLOG_v=2  # Log message at or above this level. 0:INFO, 1:WARNING, 2:ERROR, 3:FATAL
export HCCL_CONNECT_TIMEOUT=6000 # the real error info in modelarts can be blocked by the timeout error if this value is larger than HCCL_EXEC_TIMEOUT!
export ASCEND_GLOBAL_LOG_LEVEL=1  # Global log message level for Ascend. Setting it to 0 can slow down the process
export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1: detail, 0: simple
export DEVICE_ID=$1  # The device id to runing training on

task_name=train_exp02_test_step_modeFalse_json
yaml_file=configs/train_exp02_motion_transfer.yaml
#yaml_file=configs/train_text_to_video.yaml
output_path=outputs
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}

# uncomment this following line for caching and loading the compiled graph, which is saved in ${output_path}/${task_name}_cache
export MS_COMPILER_CACHE_ENABLE=0
mkdir -p ${output_path:?}/${task_name:?}_cache
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}_cache

nohup python -u train.py  \
     -c $yaml_file  \
     --output_dir $output_path/$task_name \
    > $output_path/$task_name/train.log 2>&1 &
