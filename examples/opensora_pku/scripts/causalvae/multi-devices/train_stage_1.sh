export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MS_ENABLE_NUMA=0
export MS_MEMORY_STATISTIC=1
export GLOG_v=2
output_dir="results/causalvae"
exp_name="stage1-25x256x256"

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=$output_dir/$exp_name/parallel_logs opensora/train/train_causalvae.py \
    --exp_name $exp_name \
    --model_name WFVAE \
    --model_config scripts/causalvae/wfvae_8dim.json \
    --train_batch_size 1 \
    --precision fp32 \
    --max_steps 100000 \
    --save_steps 2000 \
    --output_dir $output_dir \
    --video_path datasets/UCF-101 \
    --data_file_path datasets/ucf101_train.csv \
    --video_num_frames 25 \
    --resolution 256 \
    --dataloader_num_workers 8 \
    --start_learning_rate 1e-5 \
    --lr_scheduler constant \
    --optim adamw \
    --betas 0.9 0.999 \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --init_loss_scale 65536 \
    --jit_level "O0" \
    --use_discriminator True \
    --use_parallel True \
    --use_ema False \
    --ema_decay 0.999 \
    --perceptual_weight 0.0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --disc_start 0 \
    --wavelet_loss \
    --wavelet_weight 0.1 \
    --print_losses
