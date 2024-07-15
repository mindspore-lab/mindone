#!/bin/bash

# List of input JSON files and output folders
input_jsons=("anno_jsons/human_images_162094.json" "anno_jsons/anytext_en_1886137.json" "anno_jsons/sam_image_11185255.json" "anno_jsons/video_pixabay_65f_601513.json" "anno_jsons/video_pexel_65f_3832666.json" "anno_jsons/video_mixkit_65f_54735.json" "anno_jsons/video_pixabay_513f_51483.json" "anno_jsons/video_mixkit_513f_1997.json" "anno_jsons/video_pexel_513f_271782.json")
output_folders=("datasets/images-t5-emb-len=300" "datasets/anytext3m-t5-emb-len=300" "datasets/sam-t5-emb-len=300" "datasets/pixabay_v2-t5-emb-len=300_65f" "datasets/pexels-t5-emb-len=300_65f" "datasets/mixkit-t5-emb-len=300_65f" "datasets/pixabay_v2-t5-emb-len=300_513f" "datasets/mixkit-t5-emb-len=300_513f" "datasets/pexels-t5-emb-len=300_513f")

# Iterate through the lists and run the Python script
for i in "${!input_jsons[@]}"; do
    python opensora/sample/sample_text_embed.py \
        --data_file_path "${input_jsons[$i]}" \
        --output_path "${output_folders[$i]}"
    wait
done
