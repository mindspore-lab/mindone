# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/eval/script/cla_ssim.sh

python opensora/eval/eval.py \
    --real_video_dir /data/xiaogeng_liu/data/video1 \
    --generated_video_dir /data/xiaogeng_liu/data/video2 \
    --batch_size 10 \
    --num_frames 9 \
    --crop_size 256 \
    --resolution 256 \
    --device 'Ascend' \
    --metric 'ssim'
