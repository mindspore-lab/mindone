export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true

# msrun --master_port=8200 --worker_num=4 --local_worker_num=4 --log_dir=logs_t5_cache  \
mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout \
    python infer_t5.py \
    --dtype=fp32 \
    --batch_size=4 \
    --csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
    --output_dir tmp_t5_emb \
    --use_parallel=True \
