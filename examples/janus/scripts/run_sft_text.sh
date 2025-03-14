lr=1e-4
end_lr=1e-5
wd=0.01
bs=8
fe=False
stage=3

python train.py \
    --model_path ckpts/Janus-Pro-1B --load_weight=True \
    --task 'text' \
    --dataset_name 'pubmedqa' \
    --data_dir 'datasets/PubMedQA' \
    --training_stage $stage \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --batch_size $bs \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --max_length=512 \
    --train_steps 10000 \
    --warmup_steps 50 \
    --ckpt_save_steps 1000 \
    --ckpt_max_keep 10 \
    --output_path outputs/stage${stage}_text_pubmed_lr${lr}_wd${wd}_bs${bs} \

    # --num_samples 20 --shuffle=False \
