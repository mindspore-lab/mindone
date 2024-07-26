export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MS_ENABLE_NUMA=0
export MS_MEMORY_STATISTIC=1
export GLOG_v=2
output_dir="results/causalvae"
exp_name="9x256x256"

msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=$output_dir/$exp_name/parallel_logs opensora/train/train_causalvae.py \
    --exp_name $exp_name \
    --batch_size 1 \
    --precision fp32 \
    --max_steps 100000 \
    --save_steps 2000 \
    --output_dir $output_dir \
    --video_path /remote-home1/dataset/data_split_tt \
    --video_num_frames 9 \
    --resolution 256 \
    --sample_rate 1 \
    --dataloader_num_workers 8 \
    --load_from_checkpoint pretrained/causal_vae_488_init.ckpt \
    --start_learning_rate 1e-5 \
    --lr_scheduler constant \
    --optim adam \
    --betas 0.5 0.9 \
    --clip_grad True \
    --weight_decay 0.0 \
    --mode 0 \
    --init_loss_scale 1 \
    --jit_level "O0" \
    --use_discriminator True \
    --use_parallel True \
