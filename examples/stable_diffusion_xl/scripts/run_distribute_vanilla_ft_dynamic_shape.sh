export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"
export MS_DEV_BOOST_INFER=1
export MS_DEV_ENABLE_KERNEL_PACKET=on
export MS_ENABLE_ACLNN=1
export MS_ENABLE_NUMA=1

msrun --work_num=8 --local_worker_num=8 --master_port=8118 --log_dir=logs_dynamic_shape --join=True --cluster_time_out=300 train.py --config configs/training/sd_xl_base_finetune_multi_aspect.yaml --weight "sd_xl_base_1.0_ms_vaefix.ckpt" --data_path ./datasets/coyo_mini --save_ckpt_interval 4000 --save_path ./logs_dynamic_shape --save_path_with_time False --max_device_memory "59GB" --is_parallel True --param_fp16 True --jit_level "O1"
