export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

output_path=outputs
task_name=chinese_art_inference/txt2img


msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_path/$task_name  \
	python text_to_image.py \
     --config "configs/v1-inference.yaml" \
     --data_path "datasets/chinese_art_blip/test/prompts.txt" \
     --output_path=$output_path/$task_name \
     --ckpt_path "models/sd_v1.5-d0ab7146.ckpt" \
     --use_parallel True
