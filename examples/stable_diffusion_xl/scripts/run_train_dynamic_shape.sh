export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"
export MS_DEV_BOOST_INFER=1
export MS_DEV_ENABLE_KERNEL_PACKET=on
export MS_ENABLE_ACLNN=1
export MS_ENABLE_NUMA=1

data_path=datasets/coyo_mini
task_name=train_dynamic_shape
save_path=outputs/$task_namess

rm -rf $save_path
mkdir -p $save_path

# train
python train.py \
    --config configs/training/sd_xl_base_finetune_multi_aspect_lora.yaml \
    --data_path $data_path \
    --weight models/sd_xl_base_1.0_ms.ckpt \
    --max_device_memory "59GB" \
    --save_path $save_path \
    --jit_level "O1" \
    > $save_path/train.log 2>&1 &
