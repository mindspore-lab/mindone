export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true

# msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=log_output  \
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/infer_t5.py\
    --dtype=fp32 \
    --batch_size=4 \
    --csv_path /path/to/video_caption.csv \
    --output_path /path/to/text_embed_folder \
    --use_parallel=True \
    --model_max_length 300     # 300 for OpenSora v1.2, 200 for OpenSora v1.1, 120 for OpenSora v1.0
