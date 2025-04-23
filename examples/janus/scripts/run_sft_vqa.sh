lr=1e-4
end_lr=1e-5
wd=0.01
bs=4
fe=False
stage=3
dataset_meta_path=YOUR_DATA_PATH
pretrained_ckpt_path=YOUR_DOWNLOADED_JANUS_CKPT_PATH

python train.py \
    --model_path ${pretrained_ckpt_path} \
    --load_weight True \
    --task 'vqa' \
    --dataset_name 'medical-vqa' \
    --vqa_data_dir ${dataset_meta_path}/medical-vqa/medical-vqa \
    --training_stage $stage \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --batch_size $bs \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --max_length 1024 \
    --train_steps 10000 \
    --warmup_steps 50 \
    --ckpt_save_steps 1000 \
    --ckpt_max_keep 2 \
    --output_path outputs/stage${stage}_vqa_medicalvqa_lr${lr}_wd${wd}_bs${bs} \

    # --num_samples 20 --shuffle=False \
