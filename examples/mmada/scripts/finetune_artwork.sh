# Revise configs/mmada_finetune_artwork.yaml Config.experiment.distributed to True and zero_stage to 2 before running this script
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
msrun --bind_core=True --worker_num=4 --local_worker_num=4 --master_port=9000 --log_dir=./parallel_logs \
python training/train_mmada_stage2.py config=configs/mmada_finetune_artwork.yaml
