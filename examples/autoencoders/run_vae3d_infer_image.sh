python infer.py --model_config configs/causal_vae_f8_t4.yaml --ckpt_path models/causal_vae_3d_f8_t4_init.ckpt \
    --expand_dim_t True \
    --data_path /home/mindocr/yx/datasets/celeba_hq_256/small_test \
    --output_path samples/causal_vae_recons \
