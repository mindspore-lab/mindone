lr=1e-4
end_lr=1e-5
wd=0.01
fe=False
stage=3

python train.py \
    --model_path ckpts/Janus-Pro-1B --load_weight=True \
    --training_stage $stage \
    --csv_path 'datasets/data_demo/jade/csvfile/image_text_en.csv' \
    --data_dir 'datasets/data_demo' \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --train_steps 5000 \
    --ckpt_save_steps 500 \
    --ckpt_max_keep 10 \
    --use_value_and_grad True \
    --output_path outputs/stage${stage}_t2i_lr${lr}_wd{$wd} \
    --num_samples 20 \
