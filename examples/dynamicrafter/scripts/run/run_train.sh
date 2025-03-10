resolution=$1
num_frames=$2
output_path=$3

if [ "${resolution}" != "512" ] && [ "${resolution}" != "1024" ]; then
    echo "ERROR! Input value of 'resolution' is invalid."
    exit 1
fi

mkdir -p $output_path

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
--data_dir path/to/video_folder \
--csv_path path/to/video_caption.csv \
--text_emb_dir path/to/text_emb_folder \
--pretrained_model_path path/to/model_${resolution}.ckpt \
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
