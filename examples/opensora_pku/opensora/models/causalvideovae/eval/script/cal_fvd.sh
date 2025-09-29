# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/eval/script/cal_fvd.sh

python opensora/eval/eval.py \
    --real_video_dir /data/xiaogeng_liu/data/video1 \
    --generated_video_dir /data/xiaogeng_liu/data/video2 \
    --batch_size 10 \
    --crop_size 64 \
    --num_frames 20 \
    --device 'Ascend' \
    --metric 'fvd' \
    --fvd_method 'styleganv'
