export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=1
export INPUT_FILE_NAME="dog2"
#dog bird cyber dog2 house toy
export TASK_NAME="canny2image"

export MODEL_DIR="./models/"

output_path=output
mkdir -p ${output_path}

nohup python controlnet_image2image.py \
    --model_config "./configs/v1_inference_contorlnet.yaml" \
    --model_ckpt $MODEL_DIR"control_sd15_canny_ms.ckpt" \
    --input_image "path/to/test_imgs/"$INPUT_FILE_NAME".png" \
    --prompt "cute toy" \
    --task_name $TASK_NAME \
    2>&1 > "output/"$TASK_NAME"_logms_id"$DEVICE_ID"_"$INPUT_FILE_NAME".txt" &
