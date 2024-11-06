
local_path=/home_host/zhy/huggingface_weights

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=outputs/parallel_logs \
python finetune_with_mindspore_trainer.py \
  --model_name $local_path/meta-llama/Meta-Llama-3-8B \
  --dataset_path $local_path/Yelp/yelp_review_full \
  --output_dir ./outputs \
  --per_device_train_batch_size 8 \
  \
  --is_distribute True \
  --zero_stage 2 \
  --fp16
