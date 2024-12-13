resolution=$1
num_frames=$2
output_path=output/$3
mkdir -p $output_path

if [ "${resolution}" != "512" ] && [ "${resolution}" != "1024" ]; then
    echo "ERROR! Input value of 'resolution' is invalid."
    exit 1
fi


if [ "$resolution" = "512" ]; then
    H=320
    W=512
elif [ "$resolution" = "1024" ]; then
    H=576
    W=1024
else
    echo "Invalid resolution input. Please enter 512 or 1024."
    exit 1
fi



python scripts/train.py \
--model_config configs/training_${resolution}_v1.0.yaml \
--data_dir /root/lhy/data/mixkit-100videos/mixkit \
--csv_path /root/lhy/data/mixkit-100videos/video_caption_train.csv \
--text_emb_dir text_emb/mixkit100 \
--pretrained_model_path /root/lhy/ckpt/dynamicrafter/ms/model_${resolution}.ckpt \
--batch_size 1 \
--num_frames ${num_frames} \
--resolution ${H} ${W} \
--max_device_memory 59GB \
--epochs 40000 \
--ckpt_save_steps 100 \
--output_path $output_path \
--amp_level O0 \
--mode 0 \
--jit_level O1 \

# --amp_dtype fp16 \
# --debug True \
# --pretrained_model_path /root/lhy/ckpt/dynamicrafter/ms/model_1024.ckpt \
# --data_dir video_data/${resolution} \
# --csv_path prompts/${resolution}/test_prompts.csv \
