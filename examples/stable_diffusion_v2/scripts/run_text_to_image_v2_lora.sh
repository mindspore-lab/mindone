export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export DEVICE_ID=7

# modify to your local data path
data_path=./datasets/pokemon_blip/test/prompts.txt
#data_path=/home/yx/datasets/diffusion/pokemon/test/test_prompts.txt
lora_ckpt_path=output/pokemon/txt2img/ckpt/rank_0/sd-72.ckpt
output_path=output/lora_pokemon
lora_ft_text_encoder=False # set True if finetuned text encoder as well

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
    --lora_ft_text_encoder $lora_ft_text_encoder \
    --lora_ckpt_path $lora_ckpt_path \
    --dpm_solver_pp \
    --sampling_steps 20 \
    --data_path $data_path \
    #--prompt "a drawing of a flying dragon" \
    #--ckpt_path models/ \
