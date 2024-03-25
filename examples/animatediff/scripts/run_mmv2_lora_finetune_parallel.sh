# Parallel config
# Please generate the rank table file via hccl_tools.py
# (https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) for your own server
# num_devices=4
# rank_table_file=/home/hyx/tools/hccl_4p_4567_127.0.0.1.json
# CANDIDATE_DEVICE=(4 5 6 7)

num_devices=8
rank_table_file=/home/hyx/tools/hccl_8p_01234567_127.0.0.1.json
CANDIDATE_DEVICE=(0 1 2 3 4 5 6 7)


export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
export RANK_TABLE_FILE=$rank_table_file
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# Training path config
output_path=outputs
task_name=dist_mmv2_lora_cfg2_FA_DS

# parallel train
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
cp $0 $output_path/.

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_path:?}/${task_name:?}/rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    python train.py --config configs/training/mmv2_lora.yaml --image_size 512 --train_batch_size 1 \
        --use_recompute=False \
        --dataset_sink_mode=True \
        --enable_flash_attention=True \
        --output_path=$output_path/$task_name \
        --use_parallel=True \
        > $output_path/$task_name/rank_$i/train.log 2>&1 &
done
