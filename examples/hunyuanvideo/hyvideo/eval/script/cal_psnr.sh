python hyvideo/eval/eval_common_metrics.py \
    --real_video_dir datasets/MCL_JCV/ \
    --generated_video_dir datasets/MCL_JCV_generated/ \
    --batch_size 10 \
    --num_frames 33 \
    --crop_size 360 \
    --resolution 640 \
    --device 'Ascend' \
    --metric 'psnr'
