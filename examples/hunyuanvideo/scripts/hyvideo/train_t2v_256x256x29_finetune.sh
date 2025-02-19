export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=8000 --log_dir="./finetune_parallel_logs" \
  scripts/train.py \
   --config configs/finetune/mixkit_256x256x29.yaml \
   --train.settings.zero_stage 3 \
