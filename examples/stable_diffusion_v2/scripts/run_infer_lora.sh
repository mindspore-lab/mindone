export DEVICE_ID=$1
export MS_ENABLE_REF_MODE=1

python text_to_image.py \
        --prompt "a painting of a tree with a mountain in the background and a person standing in the foreground with a snow covered ground" \
        --use_lora True \
        --version "1.5" \
        --lora_ckpt_path $2
