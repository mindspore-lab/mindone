python scripts/inference.py \
    --device_target GPU \
    --bs 1 \
    --height 256 \
    --width 256 \
    --prompt_dir prompts/1024/ \
    --config configs/inference_256_v1.0.yaml \
    --savedir results/ \
    --mode 1 \
    --n_samples 1 \
    --unconditional_guidance_scale 7.5 \
    --ddim_steps 50 \
    --ddim_eta 1.0 \
    --text_input \
    --video_length 16 \
    --frame_stride 3 \
    # --timestep_spacing 'uniform_trailing' \
    # --guidance_rescale 0.7 \
    # --perframe_ae \
