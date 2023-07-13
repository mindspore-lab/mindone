export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export SD_VERSION="2.0" # TODO: parse by args. or fix to 2.0 later

output_path=output/finetune_pokemon
task_name=txt2img
#data_path=./datasets/diffusion/pokemon_blip/train
data_path=/home/yx/datasets/diffusion/pokemon
pretrained_model_path=models/
pretrained_model_file=sd_v2_base-57526ee4.ckpt
train_config_file=configs/train_config_v2.json
image_size=512 
train_batch_size=3 
ckpt_save_interval=1
# uncomment the following two lines to finetune on 768x768 resolution.
#image_size=768 # v2-base 512, v2.1 768
#train_batch_size=1  # 1 for 768x768, 30GB memory

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export DEVICE_NUM=8
export RANK_SIZE=8
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}
#nohup python -u run_train.py \
mpirun --allow-run-as-root -n 8 python train_text_to_image.py \
    --data_path=$data_path \
    --train_config=$train_config_file \
    --output_path=$output_path/$task_name \
    --use_parallel=False \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --image_size=$image_size \
    --train_batch_size=$train_batch_size \
    --ckpt_save_interval=$ckpt_save_interval \
#    > $output_path/$task_name/log_train_v2 2>&1 &
