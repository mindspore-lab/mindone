export DEVICE_ID=$1

# for non-INFNAN, keep drop overflow update False
# export MS_ASCEND_CHECK_OVERFLOW_MODE=1
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE" # debuggin


task_name=train_lora_sdv1 #rewrite
output_path=outputs
output_dir=$output_path/$task_name

rm -rf $output_dir
mkdir -p $output_dir
python train_text_to_image.py \
    --train_config "configs/train/train_config_lora_v1.yaml" \
    --data_path "datasets/chinese_art_blip/train" \
    --output_path $output_dir \
    --pretrained_model_path "models/sd_v1.5-d0ab7146.ckpt" \
    --loss_scaler_type "dynamic" \
    --init_loss_scale 65536 \
    --enable_flash_attention=False \
    --drop_overflow_update=True \
    --use_ema=True \
    --lora_rank=4 \
    --epochs=200 \
    --ckpt_save_interval=20 \
    > $output_dir/train.log 2>&1 &
