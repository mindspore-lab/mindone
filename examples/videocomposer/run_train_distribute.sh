output_dir='outputs/train'

# Parallel config
num_devices=8
rank_table_file=/home/huangyongxiang/tools/hccl_8p_01234567_8.92.9.90.json
CANDIDATE_DEVICE=(0 1 2 3 4 5 6 7 8)

# ascend config
export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=6000
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
RANK_TABLE_FILE=$rank_table_file
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

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
        --output_dir=$output_dir \
        --use_parallel=True \
        > $output_dir/rank_$i/train.log 2>&1 &
done
