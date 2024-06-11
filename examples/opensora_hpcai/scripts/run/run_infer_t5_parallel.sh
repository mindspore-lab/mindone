export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true

# msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=log_output  \
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/infer_t5.py\
    --dtype=fp32 \
    --batch_size=4 \
    --csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
    --output_path datasets/sora_overfitting_dataset_0410_t5 \
    --use_parallel=True \
