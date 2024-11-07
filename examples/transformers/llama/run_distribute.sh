
PORT=9000

if [ $# == 1 ]
then
    PORT=$1
fi

local_path=/home_host/zhy/huggingface_weights

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=$PORT --log_dir=outputs/parallel_logs \
python finetune_with_mindspore_trainer.py \
  --model_path $local_path/meta-llama/Meta-Llama-3-8B \
  --dataset_path $local_path/yelp_review_full \
  --output_dir ./outputs \
  \
  --zero_stage 2 \
  --bf16 \
  --is_distribute True
