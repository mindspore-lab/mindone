export HF_HUB_OFFLINE=1

output_dir=samples/$(date +"%Y.%m.%d-%H.%M.%S")

DEVICE_ID=0	python scripts/v2.0/inference_v2.py \
	--config=configs/opensora-v2-0/inference/256px.yaml \
	--prompts.prompts="A cat walks on the grass, realistic style." \
	--prompts.neg_prompts="Overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" \
	--sampling_option.num_frames=129 \
  --saving_option.output_path="$output_dir"
