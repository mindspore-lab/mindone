export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=4
export INPUT_FILE_NAME="dog2"
#dog bird cyber dog2 house toy
export TASK_NAME="canny2image"
export MODEL_DIR="./models/"


rm -rf "output/"${TASK_NAME:?}
mkdir -p "output/"${TASK_NAME:?}
nohup python image2image.py \
    --model_config "./configs/v1_inference_contorlnet.yaml" \
    --model_ckpt $MODEL_DIR"control_sd15_canny_ms.ckpt" \
    --input_image "path/to/test_imgs/"$INPUT_FILE_NAME".png" \
    --prompt "cute dog, masterpiece" \
    --image_resolution 512 \
    --task_name $TASK_NAME \
    --log_level logging.INFO \
    > "output/"$TASK_NAME"/logms_id"$DEVICE_ID"_"$INPUT_FILE_NAME"" 2>&1 &

