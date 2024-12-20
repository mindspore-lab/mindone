python infer.py \
    --model_config configs/autoencoder_kl_f8.yaml \
    --data_path datasets/celeba_hq_256/test \
    --output_path samples/vae_recons_e51 \
    --size 256 \
    --crop_size 256 \
    --mode 1 \
    --ckpt_path outputs/vae_celeba_train/ckpt/vae_kl-e51.ckpt \
    --eval_loss=True \
