export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# msrun --master_port=8200 --worker_num=8 --local_worker_num=4 --log_dir=logs_t5_cache  \
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
python scripts/infer_vae.py \
    --csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
    --video_folder datasets/sora_overfitting_dataset_0410 \
    --output_path datasets/sora_overfitting_dataset_0410_vae_512x512 \
    --vae_checkpoint models/sd-vae-ft-ema.ckpt \
    --image_size 512 \
    --vae_micro_batch_size 64 \
    --dl_return_all_frames=True \
    --use_parallel=True \
