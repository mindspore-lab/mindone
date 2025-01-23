# plot memory usage and compile info
# export MS_DEV_RUNTIME_CONF="memory_statistics:True,compile_statistics:True"

python sample_video.py \
    --ms-mode 1 \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed-type 'fixed' \
    --seed 1 \
    --save-path ./results \
    --precision 'bf16' \
    --attn-mode 'flash' \
    --use-conv2d-patchify=True \
    --model  "HYVideo-T/2-cfgdistill" \
    --dit-weight "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
    --vae-tiling=True \
    --text-embed-path 'pt_io/text_embed_with_neg-A-cat-wa.npz' \

    # --latent-noise-path 'debug/latent_noise_544x960x25.npy' \

    # --video-size 720 1280 \
    # --video-length 129 \
    # --output-type 'latent' \
    # --video-size 240 320 \
    # --video-length 17 \
    # --model  "HYVideo-T/2-depth1" \
    # --dit-weight 'ckpts/transformer_depth1.pt' \
    # --output-type 'latent' \
