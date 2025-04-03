export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="./results/sp_parallel_logs/" \
   sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --seed-type 'fixed' \
    --seed 1 \
    --save-path ./results \
    --precision 'bf16' \
    --use-parallel True \
    --sp-size 8 \
    --use-conv2d-patchify=True \
    --model  "HYVideo-T/2-cfgdistill" \
