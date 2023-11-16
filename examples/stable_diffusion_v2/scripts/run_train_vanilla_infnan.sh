export DEVICE_ID=$1

# for non-INFNAN, keep drop overflow update False
# export MS_ASCEND_CHECK_OVERFLOW_MODE=1
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE" # debug

task_name=train_vanilla_v1_infnan_moreFp32_AdamDasFusionOff
output_path=outputs
output_dir=$output_path/$task_name

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}


python train_text_to_image.py \
    --train_config "configs/train/train_config_vanilla_v1.yaml" \
    --data_path "datasets/chinese_art_blip/train" \
    --output_path $output_dir \
    --pretrained_model_path "models/sd_v1.5-d0ab7146.ckpt" \
    --init_loss_scale 65536. \
    --loss_scaler_type "dynamic" \
    --enable_flash_attention=False \
    --drop_overflow_update=True \
    > $output_dir/train.log 2>&1 &
