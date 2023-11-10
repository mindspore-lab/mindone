export DEVICE_ID=$1

# for non-INFNAN, keep drop overflow update False
# export MS_ASCEND_CHECK_OVERFLOW_MODE=1
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE" # debuggin


task_name=train_lora_ovfDropUpdate_ls16384_infnan_noema #rewrite
output_path=outputs
output_dir=$output_path/$task_name

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}

python train_text_to_image.py \
    --train_config "configs/train/train_config_lora_v2.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --output_path $output_dir \
    --pretrained_model_path "models/sd_v2_base-57526ee4.ckpt" \
    --loss_scaler_type "dynamic" \
    --init_loss_scale 16384 \
    --enable_flash_attention=False \
    --drop_overflow_update=True \
    --use_ema=False \
    > $output_dir/train.log 2>&1 &
