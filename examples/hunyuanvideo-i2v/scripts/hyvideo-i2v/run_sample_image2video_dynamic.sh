#!/bin/bash

GITHUB_BASE_URL="https://raw.githubusercontent.com/Tencent/HunyuanVideo-I2V/2dedfdc9e96c149df63c9a3b39d97e7ac290b8c1"
FILE_PATH="./assets/demo/i2v/imgs/0.jpg"
RELATIVE_PATH="${FILE_PATH#./}"
GITHUB_URL="$GITHUB_BASE_URL/$RELATIVE_PATH"

if [ ! -f "$FILE_PATH" ]; then
    echo "$FILE_PATH does not exist, downloading it from github..."
    mkdir -p "$(dirname "$FILE_PATH")"
    if curl -k -o "$FILE_PATH" "$GITHUB_URL"; then
        echo "$FILE_PATH downloaded successfully."
    else
        echo "Failed to download $FILE_PATH from GitHub. Please check the URL or your internet connection, or manually download it from torch repository."
    fi
else
    echo "$FILE_PATH already exists."
fi


python3 sample_image2video.py \
    --prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \
    --i2v-image-path $FILE_PATH \
    --model HYVideo-T/2 \
    --i2v-mode \
    --i2v-resolution 720p \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 17.0 \
    --embedded-cfg-scale 6.0 \
    --seed 0 \
    --save-path ./results \

# More example
#    --prompt "A girl walks on the road, shooting stars pass by." \
#    --i2v-image-path ./assets/demo/i2v/imgs/1.png \
