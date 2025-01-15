export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="parallel_logs/" examples/rec_video_folder.py \
    --batch_size 1 \
    --real_video_dir datasets/UCF-101/ \
    --data_file_path datasets/ucf101_test.csv \
    --generated_video_dir recons/ucf101_test/ \
    --device Ascend \
    --sample_fps 30 \
    --sample_rate 1 \
    --num_frames 25 \
    --height 256 \
    --width 256 \
    --num_workers 8 \
    --ae "WFVAEModel_D8_4x8x8" \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --use_parallel True \
    # --ms_checkpoint path/to/ms/ckpt
