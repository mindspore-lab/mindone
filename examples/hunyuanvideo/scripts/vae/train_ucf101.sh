export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=6000 --log_dir="./parallel_logs" \
 scripts/train_vae.py  \
 --config configs/vae/train/ucf101_256x256x49.yaml \
