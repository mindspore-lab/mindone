
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="./sample_videos/parallel_logs/" \
   sample_video.py \
    --video-size 256 256 \
    --video-length 29 \
    --infer-steps 50 \
    --prompt "The video features a black and white dog sitting upright against a plain background." \
    --flow-reverse \
    --seed-type 'fixed' \
    --seed 1 \
    --save-path ./sample_videos \
    --use_parallel True \
    --dit-weight path/to/ms/ckpt/ \
    --zero-stage 3 \
