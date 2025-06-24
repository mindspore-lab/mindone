# Revise configs/mmada_pretraining_stage1_llada_instruct.yaml Config.experiment.distributed to True before running this script
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
msrun --bind_core=True --worker_num=4 --local_worker_num=4 --master_port=9000 --log_dir=./parallel_logs \
python training/train_mmada.py config=configs/mmada_pretraining_stage1_llada_instruct.yaml
