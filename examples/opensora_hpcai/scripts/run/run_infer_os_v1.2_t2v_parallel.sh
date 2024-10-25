export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/inference.py --config configs/opensora-v1-2/inference/sample_t2v.yaml \
        --ckpt_path /path/to/STDiT3-s[NUM].ckpt \
        --prompt_path /path/to/video_caption.csv \
        --image_size 384 672 \
        --num_frames 204 \
        --use_parallel=True \

        # --dtype bf16 \
