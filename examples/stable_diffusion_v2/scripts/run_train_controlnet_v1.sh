export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export SD_VERSION="1.x"
device_id=6

output_path=output/
task_name=train_controlnet_all_sigmoid_091142053
data_path=/home/mindspore/congw/data/pokemon_blip_canny/train
pretrained_model_path=/home/mindspore/congw/data/
pretrained_model_file=control_sd15_canny_ms.ckpt
train_config_file=configs/train_controlnet_sd_v1.json

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}; \
# python train_controlnet.py \

nohup python -u train_controlnet.py \
    --data_path=$data_path \
    --train_config=$train_config_file \
    --output_path=$output_path/$task_name \
    --use_parallel=False \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --train_controlnet True \
   > $output_path/$task_name/log_train 2>&1 &