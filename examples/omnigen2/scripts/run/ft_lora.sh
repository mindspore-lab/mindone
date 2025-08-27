export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

output_dir=experiments/ft/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8210 --worker_num=8 --local_worker_num=8 --log_dir="$output_dir" \
python scripts/train.py \
  --config=configs/finetune/ft_lora.yml \
  --env.distributed True
