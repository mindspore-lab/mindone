unset RANK_TABLE_FILE
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# enable kbk, memory efficient
# export MS_ENABLE_ACLNN=1
# export GRAPH_OP_RUN=1

# msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=log_output  \
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
    python scripts/inference.py \
    --config configs/opensora/inference/stdit_512x512x64.yaml \
    --ckpt_path /path/to/STDiT-e[NUM].ckpt \
    --prompt_path /path/to/prompt.csv \
    --use_parallel=True \
