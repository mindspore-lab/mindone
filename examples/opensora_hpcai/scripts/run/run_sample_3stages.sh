# modify the following paths your task
config=configs/opensora/inference/stdit_512x512x64.yaml
ckpt_path=outputs/trained_stdit.ckpt
prompt_path=assets/texts/t2v_samples.txt
device_target=Ascend

# config=configs/opensora/inference/stdit_256x256x16.yaml
# ckpt_path=models/models/OpenSora-v1-HQ-16x256x256.ckpt
# device_target=GPU

rm -rf samples/t5_embed
rm -rf samples/denoised_latents

python scripts/infer_t5.py --config $config \
    --prompt_path $prompt_path \
    --output_path  samples/t5_embed \
    --device_target $device_target \


python scripts/inference.py --config $config \
    --text_embed_folder samples/t5_embed \
    --ckpt_path $ckpt_path \
    --use_vae_decode=False \
    --save_latent=True \
    --dtype=fp16 \
    --enable_flash_attention=True \
    --device_target $device_target \


python scripts/infer_vae_decode.py --fps=12 \
    --latent_folder samples/denoised_latents \
    --device_target $device_target \
