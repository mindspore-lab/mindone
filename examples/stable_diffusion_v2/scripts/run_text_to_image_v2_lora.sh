export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export DEVICE_ID=0

# modify to your local data path
data_path=./datasets/pokemon_blip/test/prompts.txt
#data_path=/home/yx/datasets/diffusion/pokemon/test/test_prompts.txt
lora_ckpt_path=output/lora_pokemon_rank4_wd1e-2/txt2img/ckpt/sd-72.ckpt
output_path=output/lora_pokemon_r4_wd1e-2_samples

n_samples=2
n_iter=1
scale=9.0

python text_to_image.py \
    --config configs/v2-inference.yaml \
    --output_path $output_path \
    --seed 42 \
    --n_iter $n_iter \
    --n_samples $n_samples \
    --scale=$scale \
    --W 512 \
    --H 512 \
    --use_lora True \
    --lora_ckpt_path $lora_ckpt_path \
    --dpm_solver \
    --sampling_steps 20 \
    --data_path $data_path \
    #--prompt "a drawing of a flying dragon" \
    #--ckpt_path models/ \
