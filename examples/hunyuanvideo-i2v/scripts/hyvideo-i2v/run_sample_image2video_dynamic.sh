#!/bin/bash

python3 sample_image2video.py \
    --prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \
    --i2v-image-path ./assets/demo/i2v/imgs/0.jpg \
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
    --ms-mode 1 \

# More example
#    --prompt "A girl walks on the road, shooting stars pass by." \
#    --i2v-image-path ./assets/demo/i2v/imgs/1.png \
