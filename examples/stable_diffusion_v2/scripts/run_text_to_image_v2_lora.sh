
export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export DEVICE_ID=4

export SD_VERSOIN="2.0"

python text_to_image.py \
    --prompt "a drawing of a flying dragon" \
    --config configs/v2-inference.yaml \
    --output_path ./output/lora_pokemon_exp4/ \
    --seed 42 \
    --n_iter 2 \
    --n_samples 4 \
    --W 512 \
    --H 512 \
    --dpm_solver \
    --ddim_steps 15 \
    --use_lora True \
    --ckpt_path output/lora_pokemon_exp4/txt2img/ckpt/rank_0 \
    --ckpt_name sd-18_208.ckpt \
