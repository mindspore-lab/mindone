export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# msrun --master_port=8200 --worker_num=8 --local_worker_num=4 --log_dir=logs_t5_cache  \
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
python scripts/infer_vae.py \
    --csv_path datasets/mixkit_tiny/sharegpt4v_tiny.csv \
    --video_folder datasets/mixkit_tiny/video \
    --output_path datasets/mixkit_tiny/vae_embed/latent_576x1024 \
    --vae_checkpoint models/sd-vae-ft-ema.ckpt \
    --image_size 576 1024 \
    --transform_name crop_resize \
    --vae_micro_batch_size 16 \
    --dl_return_all_frames=False \
    --use_parallel=True \
