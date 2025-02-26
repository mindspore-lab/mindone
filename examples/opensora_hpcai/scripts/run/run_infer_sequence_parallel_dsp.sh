export ASCEND_RT_VISIBLE_DEVICES=0,1

msrun --master_port=8200 --worker_num=2 --local_worker_num=2 --log_dir="logs" scripts/inference.py \
    -c configs/opensora-v1-2/inference/sample_t2v.yaml \
    --use_parallel True \
    --enable_sequence_parallelism True \
    --num_frames 16s \
    --resolution 720p \
    --mode 0 \
    --jit_level O0 \
    --t5_dtype fp16 \
    --dsp True
