export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=1

model_config="./configs/v1_inference_contorlnet.yaml"
model_dir="./models/"
model_name="control_sd15_segmentation_ms.ckpt"
condition_model_name="deeplabv3plus_s16_ascend_v190_voc2012_research_cv_s16acc79.06_s16multiscale79.96_s16multiscaleflip80.12.ckpt"
data_dir="./dataset/test_imgs/"
input_file_name="bird"
#dog bird cyber dog2 house toy
input_file_type=".png"
output_path="output/"
task_name="seg2image"


rm -rf ${output_path:?}${task_name:?}
mkdir -p ${output_path:?}${task_name:?}

nohup python controlnet_image2image.py \
    --model_config $model_config \
    --model_ckpt $model_dir$model_name  \
    --input_image $data_dir$input_file_name$input_file_type \
    --scale 9.0 \
    --prompt "Bird" \
    --a_prompt "best quality, extremely detailed" \
    --mode "segmentation" \
    --image_resolution 512 \
    --condition_ckpt_path $model_dir$condition_model_name \
    --task_name $task_name \
    --output_path $output_path \
    > $output_path$task_name"/logms_id"$DEVICE_ID"_"$input_file_name 2>&1 &
