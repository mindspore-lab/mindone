export DEVICE_ID=$1

export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"
export MS_DEV_JIT_SYNTAX_LEVEL=0
export MS_DEV_BOOST_INFER=1
export MS_ENABLE_ACLNN=1
export MS_ENABLE_NUMA=1
export GRAPH_OP_RUN=1

#data_path=datasets/chinese_art_blip/train
data_path=datasets/pokemon_blip/train
task_name=train_lora_pokemon_r4_fixedko
save_path=outputs/$task_namess

rm -rf $save_path
mkdir -p $save_path

# train
python train.py \
    --config configs/training/sd_xl_base_finetune_multi_aspect.yaml \
    --data_path $data_path \
    --weight models/sd_xl_base_1.0_ms.ckpt \
    --max_device_memory "59GB" \
    --save_path $save_path \
    --dynamic_shape True
    > $save_path/train.log 2>&1 &