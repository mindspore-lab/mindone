export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

device_id=4

output_path=output/lora_pokemon_r128_ema
task_name=txt2img
data_path=/home/yx/datasets/diffusion/pokemon
pretrained_model_path=models/
pretrained_model_file=stablediffusionv2_512.ckpt
train_config_file=configs/train_config_v2.json
image_size=512
train_batch_size=4      # bs=1, grad_accu=4, in diffuser
lora_rank=128
start_learning_rate=1e-4 #lr=1e-4, lr_min =0. in diffuser
end_learning_rate=0
warmup_steps=0 # ~3 epoch. diffuser 0
epochs=72 #15000 steps=>18 epochs in diffuser
use_ema=True
clip_grad=True
max_grad_norm=1.    # 1.0 in duffuser.
ckpt_save_interval=5 # save every two epochs

# uncomment the following two lines to finetune on 768x768 resolution.
#image_size=768 # v2-base 512, v2.1 768
#train_batch_size=1  # 1 for 768x768, 30GB memory

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export RANK_SIZE=1;export DEVICE_ID=$device_id;
#export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}; \ # TODO: not support with EvalSaveCallback
#nohup python -u run_train.py \
python train_text_to_image.py \
    --data_path=$data_path \
    --train_config=$train_config_file \
    --output_path=$output_path/$task_name \
    --use_parallel=False \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --image_size=$image_size \
    --train_batch_size=$train_batch_size \
    --epochs=$epochs \
    --use_lora=True \
    --lora_rank=$lora_rank \
    --start_learning_rate=$start_learning_rate \
    --end_learning_rate=$end_learning_rate \
    --warmup_steps=$warmup_steps \
    --ckpt_save_interval=$ckpt_save_interval\
    --use_ema=$use_ema \
    --clip_grad=$clip_grad \
    --max_grad_norm=$max_grad_norm \
#    > $output_path/$task_name/log_train_v2 2>&1 &
