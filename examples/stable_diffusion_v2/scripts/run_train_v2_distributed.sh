# SD Parallel training via HCCL
# Usage: bash scripts/run_train_v2_distributed.sh

# Parallel config
use_parallel=True
num_devices=4
## Please generate the rank table file via hccl_tools.py for your own server
rank_table_file=/home/yx/tools/hccl_4p_0123_127.0.0.1.json
CANDIDATE_DEVICE=(0 1 2 3)

# Training path config
data_path=/data3/datasets/laion_art_filtered
output_path=output/sd2.1_base_laion
ckpt_save_interval=1

task_name=txt2img
pretrained_model_path=models/
pretrained_model_file=sd_v2_base-57526ee4.ckpt
train_config_file=configs/train_config_v2.json

# Hyper-param config
train_batch_size=3
start_learning_rate=1e-5
end_learning_rate=0
warmup_steps=10000
epochs=2 # TODO: reduce for larger laion dataset
use_ema=True
clip_grad=False # TODO: confirm
max_grad_norm=1.

## For phase 1 in training sd 2.0-base, images with resolution < 256x256 are filtered out. For phase 2, 512x512
## For sd2.1-base, images with resolution < 512x512 are filtered out
image_filter_size=512 # TODO: confirm
image_size=512

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

# parallel train
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
#export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_path:?}/${task_name:?}/rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    nohup python -u train_text_to_image.py \
        --data_path=$data_path \
        --image_filter_size=$image_filter_size \
        --train_config=$train_config_file \
        --output_path=$output_path/$task_name \
        --use_parallel=$use_parallel \
        --pretrained_model_path=$pretrained_model_path \
        --pretrained_model_file=$pretrained_model_file \
        --image_size=$image_size \
        --train_batch_size=$train_batch_size \
        --epochs=$epochs \
        --start_learning_rate=$start_learning_rate \
        --end_learning_rate=$end_learning_rate \
        --warmup_steps=$warmup_steps \
        --ckpt_save_interval=$ckpt_save_interval\
        --use_ema=$use_ema \
        --clip_grad=$clip_grad \
        --max_grad_norm=$max_grad_norm \
        > $output_path/$task_name/rank_$i/train.log 2>&1 &
done
