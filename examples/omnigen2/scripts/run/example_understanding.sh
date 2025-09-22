#!/bin/bash

python inference_chat.py \
--model_path "OmniGen2/OmniGen2" \
--chat_mode \
--instruction "Please describe this image briefly." \
--input_image_path example_images/02.jpg
