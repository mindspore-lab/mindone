export MS_ASCEND_CHECK_OVERFLOW_MODE=1

python train_text_to_image.py \
    --train_config "configs/train/train_config_lora_v1.yaml" \
    --data_path "datasets/chinese_art_blip/train" \
    --output_path "output/lora_cnart/txt2img" \
    --pretrained_model_path "models/sd_v1.5-d0ab7146.ckpt" \
    --init_loss_scale 65536 \
