lr=1e-4
end_lr=1e-5
wd=0.1
bs=8
fe=False
stage=3
npp=0.05

python train.py \
    --model_path ckpts/Janus-Pro-1B --load_weight=True \
    --training_stage $stage \
    --task "t2i" \
    --csv_path 'datasets/artwork/jade/csvfile/image_text_en.csv' \
    --data_dir 'datasets/artwork' \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --batch_size $bs \
    --null_prompt_prob $npp \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --train_steps 10000 \
    --warmup_steps 50 \
    --ckpt_save_steps 1000 \
    --ckpt_max_keep 10 \
    --output_path outputs/stage${stage}_t2i_dsJade_lr${lr}_wd${wd}_bs${bs}_npp${npp} \

    # --num_samples 20 --shuffle=False \
