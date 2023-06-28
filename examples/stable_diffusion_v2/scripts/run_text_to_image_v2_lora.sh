
export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export DEVICE_ID=7

export SD_VERSOIN="2.0"

# TODO: get lora rank from checkpoint append_dict 

n_samples=2
n_iter=1

python text_to_image.py \
    --config configs/v2-inference.yaml \
    --output_path output/lora_pokemon_rank128_e72 \
    --seed 42 \
    --n_iter $n_iter \
    --n_samples $n_samples \
    --W 512 \
    --H 512 \
    --dpm_solver \
    --ddim_steps 15 \
    --use_lora True \
    --lora_rank 128 \
    --lora_ckpt_path output/lora_pokemon_rank128_e72/txt2img/ckpt/rank_0/sd-72.ckpt \
    --data_path /home/yx/datasets/diffusion/pokemon/test/test_prompts.txt \
    #--prompt "a drawing of a flying dragon" \
    #--ckpt_path models/ \
