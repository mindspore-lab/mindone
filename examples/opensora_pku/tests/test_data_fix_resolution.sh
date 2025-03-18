# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir="./parallel_logs/" \
python tests/test_data.py \
    --model OpenSoraT2V_v1_3-2B/122 \
    --text_encoder_name_1 google/mt5-xxl \
    --dataset t2v \
    --num_frames 93 \
    --data "scripts/train_data/video_data_v1_2.txt" \
    --cache_dir "./" \
    --ae WFVAEModel_D8_4x8x8 \
    --ae_path "LanguageBind/Open-Sora-Plan-v1.3.0/vae" \
    --sample_rate 1 \
    --max_height 352 \
    --max_width 640 \
    --force_resolution \
    --train_fps 16 \
    --interpolation_scale_t 1.0 \
    --interpolation_scale_h 1.0 \
    --interpolation_scale_w 1.0 \
    --train_batch_size=1 \
    --dataloader_num_workers 8 \
    --output_dir="test_data/" \
    --model_max_length 512 \
    --hw_stride 32 \
    --trained_data_global_step 0 \
    --group_data \
    --dataset_sink_mode False \
    # --use_parallel True \
