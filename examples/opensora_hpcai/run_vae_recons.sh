python scripts/inference_vae.py \
	--ckpt_path models/OpenSora-VAE-v1.2/model.ckpt \
    --use_temporal_vae=True \
    --image_size 256 \
    --num_frames 32 \
    --dtype bf16 \
    --video_folder ../videocomposer/datasets/webvid5 \

    # --ckpt_path outputs/vae_stage2.ckpt \
    # --ckpt_path outputs/vae_1p_stage3_kbk_bf16/ckpt/vae_3d-e400.ckpt \
	# --device_target GPU \
    # --crop_size 256 \
	# --ckpt_path /home/mindocr/yx/mindone/examples/opensora_hpcai/models/v1.2/vae.ckpt \
	# --ckpt_path /home/mindocr/yx/mindone/examples/opensora_hpcai/models/sd-vae-ft-ema.ckpt \
