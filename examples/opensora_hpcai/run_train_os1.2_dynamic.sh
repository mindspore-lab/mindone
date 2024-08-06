export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0
export MS_DATASET_SINK_QUEUE=4

# dynamic shape acceleration
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

output_dir=outputs/test_OSv1.2_kbk_dynamic

# --sample_method uniform \

python scripts/train.py \
--vae_micro_batch_size=4 \
--vae_micro_frame_size=17 \
--pretrained_model_path="" \
--mode=0 \
--jit_level O1 \
--config configs/opensora-v1-2/train/train_stage1_small.yaml \
--csv_path datasets/mixkit-100videos/video_caption_train.csv \
--video_folder datasets/mixkit-100videos/mixkit \
--text_embed_folder  datasets/mixkit-100videos/t5_emb_300 \
--dataset_sink_mode=False \
--num_parallel_workers=4 \
--prefetch_size=4 \
--enable_flash_attention=True \
--gradient_accumulation_steps=1 \
--use_ema=False \
--output_path=$output_dir \
--use_recompute=False \
--vae_dtype=fp16  # FIXME: switch to bf16 when AMP issue is fixed

# --config configs/opensora-v1-2/train/train_720x1280x51.yaml \
