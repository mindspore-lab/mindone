output_dir=$1
ckpt_path=$2

# Generate t5 embedding
python infer_t5.py \
    --config configs/inference/stdit_512x512x64.yaml \
    --output_path ${output_dir}/t5_embed.npz \

# Generate latent
python sample_t2v.py \
    --config configs/inference/stdit_512x512x64.yaml \
    --embed_path ${output_dir}/t5_embed.npz \
    --use_vae_decode=False \
    --dtype=fp16 \
    --amp_level="O2" \
    --enable_flash_attention=True \
    --checkpoint ${ckpt_path} \
    --latent_save_dir ${output_dir} \

# Decode latent to generate video
python infer_vae_decode.py \
    --fps=12 \
    --video_save_dir ${output_dir} \
    --latent_path  \
     ${output_dir}/denoised_latent_00.npy \
     ${output_dir}/denoised_latent_01.npy \
     ${output_dir}/denoised_latent_02.npy \
     ${output_dir}/denoised_latent_03.npy \
     ${output_dir}/denoised_latent_04.npy \
