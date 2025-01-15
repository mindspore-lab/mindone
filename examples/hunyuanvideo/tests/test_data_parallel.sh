export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="./parallel_logs/" \
  tests/test_data.py \
    --dataset t2v \
    --data "train_data/mixkit.txt" \
    --sample_rate 1 \
    --num_frames 29 \
    --max_height 256 \
    --max_width 256 \
    --train_batch_size 1 \
    --dataloader_num_workers 1 \
    --seed=10 \
    --output_dir="t2v-video3d-29x256p/" \
    --cfg 0.1 \
    --speed_factor 1.0 \
    --drop_short_ratio 1.0 \
    --force_resolution \
    --use_parallel True \
