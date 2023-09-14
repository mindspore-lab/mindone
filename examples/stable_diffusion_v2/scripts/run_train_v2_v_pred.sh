export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

device_id=6

#output_path=output/vpred_vanilla_finetune_chinese_art_0720
output_path=output/vpred_vanilla_finetune_pokemon_0720
task_name=txt2img
data_path=./datasets/pokemon_blip/train
#data_path=/home/yx/datasets/diffusion/pokemon_blip/train
pretrained_model_path=models/
#pretrained_model_file=sd_v2_768_v-e12e3a9b.ckpt
pretrained_model_file=sd_v2-1_768_v-061732d1.ckpt
train_config_file=configs/train_config_v2.json
model_config=configs/v2-vpred-train.yaml
#image_size=512
#train_batch_size=3
# uncomment the following two lines to finetune on 768x768 resolution.
image_size=768 # v2-base 512, v2.1 768
train_batch_size=1  # 1 for 768x768, 30GB memory

echo "dataset path: $data_path"
echo "output_path: $output_path"
echo "pretrained_model_file $pretrained_model_file"

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}; \
#nohup python -u run_train.py \

# make sure [model.prediction_type: "v"] in config yaml
python train_text_to_image.py \
    --data_path=$data_path \
    --train_config=$train_config_file \
    --model_config=$model_config \
    --output_path=$output_path/$task_name \
    --use_parallel=False \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --image_size=$image_size \
    --train_batch_size=$train_batch_size \
#    > $output_path/$task_name/log_train_v2 2>&1 &
