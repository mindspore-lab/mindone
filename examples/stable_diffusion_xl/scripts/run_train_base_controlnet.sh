nohup mpirun -n 8 --allow-run-as-root python train_controlnet.py \
    --data_path DATA_PATH \
    --weight PATH TO/sd_xl_base_1.0_ms_controlnet_init.ckpt \
    --config configs/training/sd_xl_base_finetune_controlnet_910b.yaml \
    --total_step 300000 \
    --per_batch_size 2 \
    --group_lr_scaler 10.0 \
    --save_ckpt_interval 10000 \
    --max_num_ckpt 5 \
    > train.log 2>&1 &
