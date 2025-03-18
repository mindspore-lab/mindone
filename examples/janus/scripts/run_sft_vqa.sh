lr=1e-4
end_lr=1e-5
wd=0.01
bs=4
fe=False
stage=3

python train.py \
    --model_path ckpts/Janus-Pro-1B --load_weight=True \
    --task 'vqa' \
    --dataset_name 'medical-vqa' \
    --data_dir 'datasets/medical-vqa' \
    --training_stage $stage \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --batch_size $bs \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --max_length=1024 \
    --train_steps 10000 \
    --warmup_steps 50 \
    --ckpt_save_steps 1000 \
    --ckpt_max_keep 10 \
    --output_path outputs/stage${stage}_vqa_medicalvqa_lr${lr}_wd${wd}_bs${bs}_clipgrad${clip_grad} \

    # --num_samples 20 --shuffle=False \
