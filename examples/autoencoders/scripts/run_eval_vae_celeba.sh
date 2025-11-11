python infer.py \
    --data_path /home/mindocr/yx/datasets/celeba_hq_256/small_test \
    --output_path samples/vae_recons_e22 \
    --size 256 \
    --crop_size 256 \
    --mode 1 \
    --ckpt_path outputs/vae_celeba_train/ckpt/vae_kl_f8-e22.ckpt \
    --eval_loss=True \
