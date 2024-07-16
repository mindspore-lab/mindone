export DEVICE_ID=0  # an integer within [0, 7], change it to set the device ID
export MS_ENABLE_NUMA=0
export MS_MEMORY_STATISTIC=1
export GLOG_v=2

# hyper-parameters
image_size=512  # the image size of frames, same to image height and image width
use_image_num=4  # to include n number of images in an input sample
num_frames=65  # to sample m frames from a single video. The total number of images： num_frames + use_image_num
model_dtype="bf16" # the data type used for mixed precision of the diffusion transformer model (LatteT2V).
amp_level="O2" # the default auto mixed precision level for LatteT2V.
enable_flash_attention="True" # whether to use MindSpore Flash Attention
batch_size=2 # training batch size
lr="2e-05" # learning rate. Default learning schedule is constant
output_dir=t2v-f$num_frames-$image_size-img$use_image_num-videovae488-$model_dtype-FA$enable_flash_attention-bs$batch_size-t5

python opensora/train/train_t2v.py \
    --pretrained LanguageBind/Open-Sora-Plan-v1.1.0/t2v.ckpt \
    --model LatteT2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path LanguageBind/Open-Sora-Plan-v1.1.0 \
    --video_data "scripts/train_data/video_data.txt" \
    --image_data "scripts/train_data/image_data.txt" \
    --num_frames $num_frames \
    --max_image_size $image_size \
    --enable_flash_attention $enable_flash_attention \
    --batch_size=$batch_size \
    --dataloader_num_workers 10 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000000 \
    --start_learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --precision=$model_dtype \
    --amp_level=$amp_level \
    --checkpointing_steps=500 \
    --output_dir=$output_dir \
    --model_max_length 300 \
    --clip_grad True \
    --use_image_num $use_image_num \
    --enable_tiling \
      --use_recompute True \
      --dataset_sink_mode True \
      --num_no_recompute 6 \
      --sink_size -1 \
