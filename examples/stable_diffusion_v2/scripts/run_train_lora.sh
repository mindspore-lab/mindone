export MS_ASCEND_CHECK_OVERFLOW_MODE=1

python train_text_to_image.py \
    --train_config "configs/train/train_config_lora_v2.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --output_path "output/lora_pokemon/txt2img" \
    --pretrained_model_path "models/sd_v2_base-57526ee4.ckpt" \
    --init_loss_scale 1048576 \
