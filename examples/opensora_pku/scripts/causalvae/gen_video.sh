export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1
python examples/rec_video_vae.py \
    --batch_size 1 \
    --real_video_dir ../test_eval/eyes_test \
    --generated_video_dir ../test_eval/eyes_gen \
    --device Ascend \
    --sample_fps 10 \
    --sample_rate 1 \
    --num_frames 65 \
    --resolution 512 \
    --crop_size 512 \
    --num_workers 8 \
    --ckpt LanguageBind/Open-Sora-Plan-v1.1.0/vae \
    --enable_tiling
