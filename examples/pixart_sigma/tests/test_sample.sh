# Test the major functionailities of the sampling script

OUTPUT_DIR="tests/output"

rm -r $OUTPUT_DIR
mkdir -p ${OUTPUT_DIR}

# single prompt
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml \
    --prompt "A small cactus with a happy face in the Sahara desert." \
    --output_path ${OUTPUT_DIR}/single_prompt 2>&1 | tee ${OUTPUT_DIR}/single_prompt.log

# multiple prompts
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece" \
    --output_path ${OUTPUT_DIR}/multiple_prompts 2>&1 | tee ${OUTPUT_DIR}/multiple_prompts.log

# multiple prompts (batch size > 1)
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece" \
    --batch_size 4 \
    --output_path ${OUTPUT_DIR}/multiple_prompts_batch_4 2>&1 | tee ${OUTPUT_DIR}/multiple_prompts_batch_4.log
