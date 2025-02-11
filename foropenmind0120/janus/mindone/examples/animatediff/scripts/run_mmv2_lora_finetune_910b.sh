export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python train.py --config configs/training/mmv2_lora.yaml

:' Uncommet the following line to train in image 512x512, using recompute
python train.py --config configs/training/mmv2_lora.yaml  \
    --image_size 512 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_path outputs/mmv2_lora_finetune_512bs1ga4  \
    --dataset_sink_mode True \
    --use_recompute True \
    --data_path ../videocomposer/datasets/webvid5 \
'
