python scripts/eval/eval_common_metrics.py \
    --real-video-dir datasets/MCL_JCV/ \
    --generated-video-dir datasets/MCL_JCV_generated/ \
    --batch-size 10 \
    --num-frames 33 \
    --height 360 \
    --width 640 \
    --short-size 360 \
    --device 'Ascend' \
    --metric 'ssim'
