export ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=./parallel_logs \
python training/train_mmada_stage2.py config=configs/mmada_finetune_artwork.yaml
