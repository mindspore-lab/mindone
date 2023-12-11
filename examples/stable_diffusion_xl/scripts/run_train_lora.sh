export DEVICE_ID=$1
export MS_PYNATIVE_GE=1
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
#export MS_ASCEND_CHECK_OVERFLOW_MODE=1 # "INFNAN_MODE"
export PYTHONPATH=$(pwd):$PYTHONPATH

#data_path=datasets/chinese_art_blip/train
data_path=datasets/pokemon_blip/train
task_name=train_lora_pokemon_r4_InfNan
save_path=outputs/$task_name

rm -rf $save_path
mkdir -p $save_path

# train
python train_lora.py \
    --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
    --data_path $data_path \
    --weight models/sd_xl_base_1.0_ms.ckpt \
    --save_path $save_path \
    > $save_path/train.log 2>&1 &

# infer
# python demo/sampling_without_streamlit.py --task txt2img --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml --weigh models/sd_xl_base_1.0_ms.ckpt $save_path/weights/SDXL-base-1.0_2000_lora.ckpt --prompt "a sks dog swimming in a pool"
