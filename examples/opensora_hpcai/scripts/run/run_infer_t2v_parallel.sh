export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# msrun --master_port=8200 --worker_num=4 --local_worker_num=4 --log_dir=logs_t5_cache  \
mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/inference.py\
    --config configs/opensora/inference/stdit_512x512x64.yaml \
    --ckpt_path outputs/stdit_vaeO2Fp16_ditBf16_rc-4_512x512x64/2024-05-09T01-45-32/ckpt/STDiT-e200.ckpt \
    --prompt_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
    --use_parallel=True \
