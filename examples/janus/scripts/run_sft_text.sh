lr=1e-4
end_lr=1e-5
wd=0.01
bs=8
fe=False
stage=3
dataset_meta_path=YOUR_DATA_PATH
pretrained_ckpt_path=YOUR_DOWNLOADED_JANUS_CKPT_PATH

python train.py \
    --ms_mode 0 \
    --model_path ${pretrained_ckpt_path} \
    --load_weight True \
    --task 'text' \
    --dataset_name 'pubmedqa' \
    --text_qa_data_dir ${dataset_meta_path}/PubMedQA/pqa_labeled \
    --training_stage $stage \
    --learning_rate $lr \
    --end_learning_rate $end_lr \
    --batch_size $bs \
    --weight_decay $wd \
    --freeze_embedding $fe \
    --max_length 512 \
    --train_steps 10000 \
    --warmup_steps 50 \
    --ckpt_save_steps 1000 \
    --ckpt_max_keep 2 \
    --output_path outputs/stage${stage}_text_pubmed_lr${lr}_wd${wd}_bs${bs} \

    # --num_samples 20 --shuffle=False \
