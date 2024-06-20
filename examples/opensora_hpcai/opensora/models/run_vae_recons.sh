
python vae_recons.py \
	--ckpt_path /home/mindocr/yx/mindone/examples/opensora_hpcai/models/v1.2/vae.ckpt \
    --use_temporal_vae=True \
    --dataset_name video \
    --size 512 \
    --crop_size 256 \
    --num_frames 32 \
    --dtype fp32 \
    --data_path /home/mindocr/yx/mindone/examples/videocomposer/datasets/webvid5 \
	--device_target GPU \

	# --ckpt_path /home/mindocr/yx/mindone/examples/opensora_hpcai/models/sd-vae-ft-ema.ckpt \
