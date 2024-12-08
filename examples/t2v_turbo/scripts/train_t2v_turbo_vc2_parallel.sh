#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
# export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
# export MS_MEMORY_STATISTIC=0

# export MS_DATASET_SINK_QUEUE=4

# operation/graph fusion for dynamic shape
# export MS_DEV_ENABLE_KERNEL_PACKET=on

# enable kbk : 1
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

# log level
export GLOG_v=2

# data paths
DATA_PATH=datasets/webvid/2M_train/part5/00000/
CSV_PATH=datasets/webvid/2M_train/part5/00000/video_caption.csv
output_dir=outputs/t2v-train/

# model weights
t2v_model_path=model_cache/t2v-vc2/VideoCrafter2-model-ms.ckpt
t2v_encoder_path=model_cache/t2v-vc2/open_clip_vit_h_14-9bb07a10.ckpt
video_rm_path=model_cache/t2v-vc2/InternVideo2-stage2_1b-224p-f4.ckpt
image_rm_path=model_cache/t2v-vc2/HPS_v2.1_compressed.ckpt


msrun --bind_core=True --master_port=8090 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
  python train_t2v_turbo_vc2.py \
    --use_parallel True \
    --pretrained_model_path $t2v_model_path \
    --pretrained_enc_path $t2v_encoder_path \
    --train_batch_size 1 \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 8 \
    --use_recompute True \
    --reward_fn_name hpsv2 \
    --reward_scale 1.0 \
    --image_rm_ckpt_dir $image_rm_path \
    --video_rm_name vi_clip2 \
    --video_reward_scale 2.0 \
    --video_rm_ckpt_dir $video_rm_path \
    --no_scale_pred_x0 \
    --csv_path $CSV_PATH \
    --data_path $DATA_PATH \
    --dataloader_num_workers 1 \
    --mode 1 \
    --cast_teacher_unet \
    --lora_rank 64 \
    --jit_level O0 \
    --mixed_precision fp16 \
    --n_frames 16 \
    --debug False \
    --reward_batch_size 5 \
    --video_rm_batch_size 8 \
    --learning_rate 1.0e-5 \
    --lr_warmup_steps 500 \
    --loss_type huber \
    --output_dir $OUTPUT_DIR \
