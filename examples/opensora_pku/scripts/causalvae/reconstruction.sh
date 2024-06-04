export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1
python examples/rec_imvi_vae.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0/vae \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --device Ascend \
    --sample_rate 1 \
    --num_frames 65 \
    --resolution 512 \
    --crop_size 512 \
    --ae CausalVAEModel_4x8x8 \
    --enable_tiling
