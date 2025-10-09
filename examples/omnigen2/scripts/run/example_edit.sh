#!/bin/bash

python inference.py \
--model_path "OmniGen2/OmniGen2" \
--num_inference_step 50 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--instruction "Change the background to classroom." \
--input_image_path example_images/ComfyUI_temp_mllvz_00071_.png \
--output_image_path outputs/output_edit.png \
--num_images_per_prompt 1
