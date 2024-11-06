export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/inference.py --config configs/opensora-v1-1/inference/sample_t2v.yaml \
        --ckpt_path /path/to/STDiT-e[NUM].ckpt \
        --prompt_path /path/to/video_caption.csv \
        --image_size 576 1024 \
        --num_frames 24 \
        --vae_micro_batch_size 8 \
        --loop 1 \
        --use_parallel=True \

        # --dtype bf16 \
