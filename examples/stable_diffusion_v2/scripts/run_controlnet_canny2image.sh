export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=7
export INPUT_FILE_NAME="toy"
#dog bird cyber dog2 house toy 
export TASK_NAME="canny2image"

export DATA_DIR="/home/mindspore/congw/data/"

nohup python image2image.py \
    --model_config "./configs/v1_inference_contorlnet.yaml" \
    --model_ckpt $DATA_DIR"control_sd15_canny_ms.ckpt" \
    --input_image $DATA_DIR"test_imgs/"$INPUT_FILE_NAME".png" \
    --prompt "cute toy" \
    --task_name $TASK_NAME \
    2>&1 > "output/"$TASK_NAME"_logms_id"$DEVICE_ID"_"$INPUT_FILE_NAME".txt" &
    