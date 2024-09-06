# Test the major functionailities of the sampling script

OUTPUT_DIR="tests/output/"

mkdir -p ${OUTPUT_DIR}

# single prompt
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml \
    --prompt "A small cactus with a happy face in the Sahara desert." \
    --output_path ${OUTPUT_DIR}/single_prompt > ${OUTPUT_DIR}/single_prompts.log 2>&1

# multiple prompts
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece" \
    --output_path ${OUTPUT_DIR}/multiple_prompts > ${OUTPUT_DIR}/multiple_prompts.log 2>&1

# single prompt (batch size == 4)
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml \
    --prompt "A small cactus with a happy face in the Sahara desert." \
    --batch_size 4 \
    --output_path ${OUTPUT_DIR}/single_prompt_batch_4 > ${OUTPUT_DIR}/single_prompt_batch_4.log 2>&1

# multiple prompts (batch size == 4)
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece" \
    --batch_size 4 \
    --output_path ${OUTPUT_DIR}/multiple_prompts_batch_4 > ${OUTPUT_DIR}/multiple_prompts_batch_4.log 2>&1
