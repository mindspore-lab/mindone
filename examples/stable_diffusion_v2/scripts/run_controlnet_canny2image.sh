export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=6

model_config="./configs/v1_inference_contorlnet.yaml"
model_dir="./models/"
model_name="control_sd15_canny_ms.ckpt"
data_dir="./dataset/test_imgs/"
input_file_name="dog"
#dog bird cyber dog2 house toy
input_file_type=".png"
output_path="output/"
task_name="canny2image"


rm -rf ${output_path:?}${task_name:?}
mkdir -p ${output_path:?}${task_name:?}

nohup python controlnet_image2image.py \
    --model_config $model_config \
    --model_ckpt $model_dir$model_name \
    --input_image $data_dir$input_file_name$input_file_type \
    --prompt "dog, masterpiece" \
    --image_resolution 512 \
    --task_name $task_name \
    --output_path $output_path \
    --log_level logging.INFO \
    > $output_path$task_name"/logms_id"$DEVICE_ID"_"$input_file_name 2>&1 &
