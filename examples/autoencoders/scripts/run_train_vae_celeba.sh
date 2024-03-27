python train.py \
    --config configs/training/vae_celeba.yaml \
    --output_path="outputs/vae_celeba_train_ema" \
    --data_path="datasets/celeba_hq_256/train" \
    --mode=0 \
    --device_target="GPU" \
