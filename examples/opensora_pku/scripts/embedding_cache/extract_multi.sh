#!/bin/bash
min=1024
max=65535
# List of input JSON files and output folders
input_json="anno_jsons/human_images_162094.json"
output_folder="datasets/images-t5-emb-len=300"
log_folder="output_log"  # the log files are saved under this folder
random_port=$((RANDOM % (max - min + 1) + min))
msrun --master_port=$random_port --worker_num=8 --local_worker_num=8 --log_dir=$log_folder opensora/sample/sample_text_embed.py \
    --data_file_path $input_json \
    --output_path $output_folder \
    --use_parallel True
