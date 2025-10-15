lr=1e-4
end_lr=1e-5
wd=0.1
bs=8
fe=False
stage=3
npp=0.05
dataset_meta_path=YOUR_DATA_PATH
pretrained_ckpt_path=YOUR_DOWNLOADED_JANUS_CKPT_PATH

python train.py \
    --model_path ${pretrained_ckpt_path} \
    --load_weight True \
    --training_stage $stage \
    --task "t2i" \
    --t2i_csv_path ${dataset_meta_path}/data_demo/jade/csvfile/image_text_en.csv \
    --t2i_data_dir ${dataset_meta_path}/data_demo \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --batch_size $bs \
    --null_prompt_prob $npp \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --max_length 1024 \
    --train_steps 10000 \
    --warmup_steps 50 \
    --ckpt_save_steps 1000 \
    --ckpt_max_keep 2 \
    --output_path outputs/stage${stage}_t2i_dsJade_lr${lr}_wd${wd}_bs${bs}_npp${npp} \

    # --num_samples 20 --shuffle=False \
