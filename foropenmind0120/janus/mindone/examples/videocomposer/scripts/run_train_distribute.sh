# export MS_ENABLE_REF_MODE=1 # NEEDED for 910B + MS2.2 0907 version for saving checkpoint correctly
export MS_ASCEND_CHECK_OVERFLOW_MODE=1 # for ms+910B, check overflow
#export INF_NAN_MODE_ENABLE=1 # For pytorch+npu, recommend to enable it for mixed precision training for 910B. it determines how overflow is detected

task_name=train_exp02_motion_transfer_latest_rmAssert_8p
yaml_file=configs/train_exp02_motion_transfer.yaml
output_path=outputs

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
# uncomment this following line for caching and loading the compiled graph, which is saved in ${output_path}/${task_name}_cache
# export MS_COMPILER_CACHE_ENABLE=1
mkdir -p ${output_path:?}/${task_name:?}_cache
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}_cache

# Parallel config
# Runnable on 1013
#num_devices=2
#rank_table_file=/home/docker_home/jason/hccl_2p_45_10.170.22.51.json
#CANDIDATE_DEVICE=(4 5)

#
num_devices=8
rank_table_file=/home/docker_home/jason/hccl_8p_01234567_10.170.22.51.json
CANDIDATE_DEVICE=(0 1 2 3 4 5 6 7)

# ascend config
#export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=6000 # the real error info in modelarts can be blocked by the timeout error if this value is larger than HCCL_EXEC_TIMEOUT!
#export ASCEND_GLOBAL_LOG_LEVEL=3
#export ASCEND_SLOG_PRINT_TO_STDOUT=0

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
RANK_TABLE_FILE=$rank_table_file
export RANK_TABLE_FILE=${RANK_TABLE_FILE}
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# remove files
output_dir=$output_path/$task_name
cp $0 $output_dir/.

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
        --output_dir=$output_dir \
        --use_parallel=True \
        > $output_dir/rank_$i/train.log 2>&1 &
done
