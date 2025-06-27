export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=./parallel_logs \
python training/train_mmada.py config=configs/mmada_pretraining_stage1_llada_instruct.yaml
