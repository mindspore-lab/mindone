export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export DEVICE_ID=1
export INPUT_FILE_NAME="bird"
#dog bird cyber dog2 house toy
export TASK_NAME="seg2image"

export MODEL_DIR="./models/"

output_path=output
mkdir -p ${output_path}

nohup python image2image.py \
    --model_config "./configs/v1_inference_contorlnet.yaml" \
    --model_ckpt $MODEL_DIR"control_sd15_segmentation_ms.ckpt" \
    --input_image "path/to/test_imgs/"$INPUT_FILE_NAME".png" \
    --scale 9.0 \
    --prompt "Bird" \
    --a_prompt "best quality, extremely detailed" \
    --mode "segmentation" \
    --image_resolution 512 \
    --segmentation_ckpt_path $MODEL_DIR"deeplabv3plus_s16_ascend_v190_voc2012_research_cv_s16acc79.06_s16multiscale79.96_s16multiscaleflip80.12.ckpt" \
    --task_name $TASK_NAME \
    2>&1 > ${output_path}"/"$TASK_NAME"_logms_id"$DEVICE_ID"_"$INPUT_FILE_NAME".txt" &
