python examples/rec_imvi_vae.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0/vae/causal_vae_488.ckpt \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --sample_rate 1 \
    --num_frames 65 \
    --sample_size 512 \
    --ae CausalVAEModel_4x8x8
