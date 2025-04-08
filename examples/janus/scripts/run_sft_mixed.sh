lr=1e-4
end_lr=1e-5
wd=0.1
bs=6
fe=False
stage=3
npp=0.05

python train.py \
    --ms_mode 0 \
    --model_path ckpts/Janus-Pro-1B \
    --load_weight True \
    --use_value_and_grad False \
    --training_stage $stage \
    --task "mixed" \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --batch_size $bs \
    --null_prompt_prob $npp \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --train_steps 10000 \
    --warmup_steps 50 \
    --ckpt_save_steps 1000 \
    --ckpt_max_keep 1 \
    --output_path outputs/stage${stage}_mixed_lr${lr}_wd${wd}_bs${bs}_npp${npp} \
    --mixed_task_rand_samp \
