export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1

output_dir=samples/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8225 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir --join=True  \
	python scripts/v2.0/inference_v2.py \
	--config=configs/opensora-v2-0/inference/768px.yaml \
  --env.distributed=True \
	--prompts.prompts="A cat walks on the grass, realistic style." \
	--prompts.neg_prompts="Overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" \
	--sampling_option.num_frames=129 \
  --saving_option.output_path="$output_dir"
