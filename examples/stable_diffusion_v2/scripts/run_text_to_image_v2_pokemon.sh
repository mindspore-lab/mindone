export DEVICE_ID=7 

#ckpt_path=models
#ckpt_name=stablediffusionv2_512.ckpt
#output_path=output/sd2b_pokemon_samples

ckpt_path=output/vft_sd2.0_pokemon_freezeclip_1p/ckpt/rank_0
ckpt_name=sd-20_277.ckpt
output_path=output/samples_pokemon_ft1p

#ckpt_path=output/vft_sd2.0_pokemon_4p/ckpt/rank_0
#ckpt_name=sd-20_69.ckpt
#output_path=output/sd2b_ft8p_pokemon_samples

# dpm solver, 15 steps.
# plms, 50 steps
python text_to_image.py \
    --data_path /home/yx/datasets/diffusion/pokemon/test/test_prompts.txt \
    --output_path $output_path \
    --config configs/v2-inference.yaml \
    --ckpt_path $ckpt_path \
    --ckpt_name $ckpt_name \
    --seed 42 \
    --n_iter 1 \
    --n_samples 2 \
    --W 512 \
    --H 512 \
    --ddim_steps 50 \
    #--dpm_solver \
    #--ddim_steps 50 \
