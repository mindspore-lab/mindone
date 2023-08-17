export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=3
export INPUT_FILE_NAME="dog"
export TASK_NAME="canny2image"


nohup python canny2image.py \
    --input_image "/home/mindspore/congw/data/test_imgs/"$INPUT_FILE_NAME".png" \
    --prompt "cute dog" \
    --task_name $TASK_NAME \
    2>&1 > "output/"$TASK_NAME"_logms_id"$DEVICE_ID"_"$INPUT_FILE_NAME".txt" &
    