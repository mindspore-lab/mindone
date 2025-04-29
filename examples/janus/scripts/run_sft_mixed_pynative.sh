lr=1e-4
end_lr=1e-5
wd=0.1
bs=6
fe=False
stage=3
npp=0.05
dataset_meta_path=YOUR_DATA_PATH
pretrained_ckpt_path=YOUR_DOWNLOADED_JANUS_CKPT_PATH
ms_mode=1

python train.py \
    --ms_mode $ms_mode \
    --model_path ${pretrained_ckpt_path} \
    --load_weight True \
    --use_value_and_grad True \
    --vqa_data_dir ${dataset_meta_path}/medical-vqa/medical-vqa \
    --text_qa_data_dir ${dataset_meta_path}/PubMedQA/pqa_labeled \
    --t2i_parquet_dir ${dataset_meta_path}/ROCO-radiology/testdata \
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
    --output_path outputs/stage${stage}_mixed_lr${lr}_wd${wd}_bs${bs}_npp${npp}_mode${ms_mode} \    --mixed_task_rand_samp \
