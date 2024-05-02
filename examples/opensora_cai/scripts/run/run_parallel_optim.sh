mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/train.py --config configs/opensora/train/stdit_512x512x64.yaml \
    --num_frames 14 \
    --use_parallel True \
    --parallel_mode optim \
