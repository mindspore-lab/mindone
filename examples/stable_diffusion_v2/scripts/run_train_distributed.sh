export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

output_path=outputs
task_name=train_txt2img
data_path=datasets/pokemon_blip/train


msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_path/$task_name  \
	python train_text_to_image.py \
        --train_config="configs/train/train_config_vanilla_v1.yaml" \
        --data_path=$data_path \
        --pretrained_model_path="models/sd_v1.5-d0ab7146.ckpt" \
        --output_path=$output_path/$task_name \
        --use_parallel=True \
        --dataset_sink_mode=True
