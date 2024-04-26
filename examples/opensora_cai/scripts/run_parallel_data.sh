mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout \
    python train_t2v.py --config configs/train/stdit_512x512x64.yaml \
    --num_frames 16 \
    --enable_flash_attention True \
    --use_parallel True \
    --parallel_mode data \
    --optim  adamw_zero1 \
    --use_recompute False \
