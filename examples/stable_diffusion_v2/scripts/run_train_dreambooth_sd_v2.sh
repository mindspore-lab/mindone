export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export SD_VERSION="2.0" # TODO: parse by args. or fix to 2.0 later

device_id=0
export RANK_SIZE=1
export DEVICE_ID=$device_id

output_path=output/
task_name=txt2img
instance_data_dir=dreambooth/dataset/dog
instance_prompt="a photo of sks dog"
class_data_dir=temp_class_images/dog
class_prompt="a photo of a dog"
pretrained_model_path=models/
pretrained_model_file=sd_v2_base-57526ee4.ckpt
train_config_file=configs/train_dreambooth_sd_v2.json
image_size=512
train_batch_size=1
# uncomment the following two lines to finetune on 768x768 resolution.
#image_size=768 # v2-base 512, v2.1 768
#train_batch_size=1  # 1 for 768x768, 30GB memory

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}; \

python train_dreambooth.py \
    --mode=0 \
    --use_parallel=False \
    --instance_data_dir=$instance_data_dir \
    --instance_prompt="$instance_prompt"  \
    --class_data_dir=$class_data_dir \
    --class_prompt="$class_prompt" \
    --train_config=$train_config_file \
    --output_path=$output_path/$task_name \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --image_size=$image_size \
    --train_batch_size=$train_batch_size \
    --epochs=8 \
    --start_learning_rate=2e-6 \
    --train_text_encoder=True \
#    > $output_path/$task_name/log_train_v2 2>&1 &
