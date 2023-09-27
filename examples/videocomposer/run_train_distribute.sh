# export MS_ENABLE_REF_MODE=1 # NEEDED for 910B + MS2.2 0907 version for saving checkpoint correctly
export INF_NAN_MODE_ENABLE=1 # recommend to enable it for mixed precision training for 910B. it determines how overflow is detected

task_name=train_exp02_motion_transfer
yaml_file=configs/${task_name}.yaml
output_path=outputs
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}

# Parallel config
num_devices=8
rank_table_file=/home/huangyongxiang/tools/hccl_8p_01234567_8.92.9.90.json
CANDIDATE_DEVICE=(0 1 2 3 4 5 6 7 8)

# ascend config
#export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=6000
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
RANK_TABLE_FILE=$rank_table_file
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# uncomment this following line for caching and loading the compiled graph
#export MS_COMPILER_CACHE_ENABLE=1
#export MS_COMPILER_CACHE_PATH=/cache

# remove files
rm -rf ${output_dir:?}
mkdir -p ${output_dir:?}
cp $0 $output_dir/.
#export MS_COMPILER_CACHE_PATH=${output_dir:?}

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_dir:?}//rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    nohup python -u train.py \
        --cfg=$yaml_file  \
        --output_dir=$output_path/$task_name \
        --use_parallel=True \
        > $output_path/$task_name/rank_$i/train.log 2>&1 &
done
