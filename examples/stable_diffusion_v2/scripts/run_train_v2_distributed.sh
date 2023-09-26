# SD Parallel training via HCCL
# Usage: bash scripts/run_train_v2_distributed.sh

# Parallel config
num_devices=4
# Please generate the rank table file via hccl_tools.py
# (https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) for your own server
rank_table_file=hccl_4p_0123_127.0.0.1.json
CANDIDATE_DEVICE=(0 1 2 3)

# ascend config
export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=6000
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
export RANK_TABLE_FILE=$rank_table_file
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# Training path config
data_path=datasets/pokemon_blip/train
output_path=output/finetune_pokemon_4p_unfreeze_txtenc
pretrained_model_path=models/sd_v2_base-57526ee4.ckpt
task_name=txt2img

# parallel train
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
cp $0 $output_path/.
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
        --version="2.0" \
        --model_config="configs/v2-train.yaml" \
        --data_path=$data_path \
        --image_filter_size=200 \
        --output_path=$output_path/$task_name \
        --use_parallel=True \
        --pretrained_model_path=$pretrained_model_path \
        --image_size=512 \
        --train_batch_size=3 \
        --optim="adamw" \
        --weight_decay=0.01 \
        --epochs=20 \
        --start_learning_rate=1e-5 \
        --end_learning_rate=0 \
        --warmup_steps=250 \
        --decay_steps=0 \
        --ckpt_save_interval=5\
        --use_ema=True \
        --clip_grad=False \
        > $output_path/$task_name/rank_$i/train.log 2>&1 &
done
