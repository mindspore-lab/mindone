export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MS_ENABLE_NUMA=0
export MS_MEMORY_STATISTIC=1

# enable kbk
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1
export GLOG_v=2

# hyper-parameters
image_size=512
use_image_num=4
num_frames=17
model_dtype="fp16"
enable_flash_attention="True"
batch_size=4
lr="2e-05"
output_dir=t2v-f$num_frames-$image_size-img$use_image_num-videovae488-$model_dtype-FA$enable_flash_attention-bs$batch_size-t5

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=$output_dir/parallel_logs opensora/train/train_t2v.py \
      --data_path /remote-home1/dataset/sharegpt4v_path_cap_64x512x512.json \
      --video_folder /remote-home1/dataset/data_split_tt \
      --text_embed_folder /path/to/text-embed-folder \
      --pretrained pretrained/t2v.ckpt \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path LanguageBind/Open-Sora-Plan-v1.0.0 \
    --sample_rate 1 \
    --num_frames $num_frames \
    --max_image_size $image_size \
    --use_recompute True \
    --enable_flash_attention $enable_flash_attention \
    --batch_size=$batch_size \
    --num_parallel_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --start_learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --precision=$model_dtype \
    --checkpointing_steps=500 \
    --output_dir=$output_dir \
    --model_max_length 300 \
    --clip_grad True \
    --use_image_num $use_image_num \
    --use_img_from_vid \
    --use_parallel True \
    --parallel_mode "optim" \
