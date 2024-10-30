unset RANK_TABLE_FILE

output_dir=outputs/os1.2_stage2_16p

# rm -rf device
mkdir -p $output_dir

echo "start training"

# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on
# log level
export GLOG_v=2
dup=""

# 循环启动Worker5到Worker8，4个Worker训练进程
for((i=8;i<16;i++));
do
    export MS_WORKER_NUM=16                    # 设置集群中Worker进程总数为8（包括其他节点进程）
    export MS_SCHED_HOST=7.242.108.66  # 设置Scheduler IP地址为节点1 IP地址
    export MS_SCHED_PORT=8123                 # 设置Scheduler端口
    export MS_ROLE=MS_WORKER                  # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                      # 设置进程id，可选
    # 启动训练脚本
    python scripts/train.py \
        --pretrained_model_path="models/OpenSora-STDiT-v3/opensora_stdit_v3.ckpt" \
        --mode=0 \
        --jit_level O1 \
        --max_device_memory 55GB \
        --config configs/opensora-v1-2/train/train_stage2_ms.yaml \
        --csv_path /home_host/datasets/client_500/vcg_40w_500_data_formatted.csv \
        --video_folder /home_host/datasets/client_500 \
        --text_embed_folder  /home_host/datasets/client_500/t5_emb_300 \
        --enable_flash_attention=True \
        --gradient_accumulation_steps=1 \
        --num_parallel_workers=2 \
        --prefetch_size=2 \
        --use_ema=True \
        --output_path=$output_dir \
        --use_recompute=True \
        --vae_dtype=fp16 \
        --train_steps=8000 --ckpt_save_steps=500 \
        --use_parallel=True \
        > $output_dir/worker_$i.log 2>&1 &

done
