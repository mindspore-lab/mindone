# SD Parallel training via HCCL

# Parallel config
num_devices=8
# Please generate the rank table file via hccl_tools.py
# (https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/hccl_tools.py) for your own server
rank_table_file=hccl_4p_01234567_127.0.0.1.json
CANDIDATE_DEVICE=(0 1 2 3 4 5 6 7)

export DEVICE_NUM=$num_devices
export RANK_SIZE=$num_devices
export RANK_TABLE_FILE=$rank_table_file
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

# Training path config
data_path=datasets/pokemon_blip/train
pretrained_model_path=models/sd_v2_base-57526ee4.ckpt

output_path=outputs
task_name=train_xt2img

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
        --train_config="configs/train/train_config_vanilla_v1.yaml" \
        --data_path=$data_path \
        --output_path=$output_path/$task_name \
        --use_parallel=True \
        > $output_path/$task_name/rank_$i/train.log 2>&1 &
done
