export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# dynamic shape acceleration
export MS_DEV_ENABLE_KERNEL_PACKET=on

# stop JIT Fallback
# export MS_DEV_JIT_SYNTAX_LEVEL=0

# export TEST_IGNORE_EXCEPTION=1
# export GLOG_v=0

# export MS_DEV_DUMP_BPROP=on

frames=33
mixed_strategy=mixed_video_random
output_dir=outputs/vae_8p_stage3_ucf101_${mixed_strategy}_f$frames
out_log=log_vae_train.log

python scripts/train_vae.py \
    --config configs/vae/train/stage3.yaml \
	--csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
	--video_folder datasets/sora_overfitting_dataset_0410 \
    --dtype bf16 \
    --output_path $output_dir \
    --jit_level O0 \
    --mode=0 \
    --image_size 256 \
    --num_frames $frames \
    --epochs 200 \
    --optim adamw_re \
    --use_recompute=True \
    --mixed_strategy $mixed_strategy \
    --micro_batch_size 4 \
    --micro_frame_size 17 \
	--pretrained_model_path="models/OpenSora-VAE-v1.2/model.ckpt" &> ${out_log}  &

tail -f ${out_log}
