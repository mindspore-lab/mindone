"""
This script is achieve video understanding for multiple videos.
Need to change `MODEL_HUB` model path, and `CLIPS` video paths accordingly.

Usage:
cd examples/transformers/qwen2-vl
python video_understanding.py
"""

import time

import numpy as np
from transformers import AutoProcessor

import mindspore as ms

from mindone.transformers import Qwen2VLForConditionalGeneration
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info

ms.set_context(mode=ms.PYNATIVE_MODE)

# 1. Load Model and Processor
start_time = time.time()

print("Loading Qwen2VLForConditionalGeneration Model")
MODEL_HUB = "Qwen/Qwen2-VL-2B-Instruct"  # NOTE: REPLACE_WITH_YOUR_MODEL_PATH
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_HUB).set_train(False)

print("Loading AutoProcessor")
# default processer
processor = AutoProcessor.from_pretrained(
    MODEL_HUB
)  # Qwen2VLProcessor (image_processor: Qwen2VLImageProcessor, tokenizer: Qwen2TokenizerFast)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(MODEL_HUB, min_pixels=min_pixels, max_pixels=max_pixels)

print("Finish loading model and processor, time elapsed: %.4fs" % (time.time() - start_time))


print("*************************************************")
print("********** Start Video Understanding *************")
print("*************************************************")

# NOTE: REPLACE with your list of video paths
CLIPS = ["demo1.mp4", "demo2.mp4"]

for video_path in CLIPS:
    # 2. Prepare Inputs
    # Messages containing a video url and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": "Describe this video in detail."},
            ],
        }
    ]

    # Preparation for inference
    # prepare text inuput
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # prepare vision input
    image_inputs, video_inputs = process_vision_info(messages)  # a list of PIL Images
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="np",
    )
    # convert input to Tensor
    for key, value in inputs.items():  # by default input numpy array or list
        if isinstance(value, np.ndarray):
            inputs[key] = ms.Tensor(value)
        elif isinstance(value, list):
            inputs[key] = ms.Tensor(value)
        if inputs[key].dtype == ms.int64:
            inputs[key] = inputs[key].to(ms.int32)  # "input_ids", "attention_mask", "image_grid_thw"

    # 3. Inference: Generation of Tokens, Decode Tokens
    start_time = time.time()

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    print(f"generated_ids length / #steps: {len(generated_ids[0])}")
    elapsed = time.time() - start_time
    print("Average speed %.4fs/step" % (elapsed / len(generated_ids[0])))

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print("Generated response, time elapsed: %.4fs" % (time.time() - start_time))

    print("*" * 50)
    print("Input: %s" % str(messages))
    print("Response:", output_text)
    print("*" * 50)

print("******** End of Video Understanding ********")
