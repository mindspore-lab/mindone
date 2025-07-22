import os
import time

import requests
from PIL import Image
from transformers import AriaProcessor

import mindspore as ms

from mindone.transformers import AriaForConditionalGeneration

ms.set_context(mode=ms.PYNATIVE_MODE)


model_id_or_path = "rhymes-ai/Aria"

start_time = time.time()
import json

# DEBUG inference by using slimmed network
from transformers.models.aria import AriaConfig

with open(os.path.join(model_id_or_path, "config.json"), "r") as json_file:
    config_json = json.load(json_file)
config_json["text_config"]["num_hidden_layers"] = 2  # testing use
config_json["text_config"]["attn_implementation"] = "flash_attention_2"
config_json["vision_config"]["num_hidden_layers"] = 2  # testing use
config_json["vision_config"]["attn_implementation"] = "eager"
config = AriaConfig(**config_json)
model = AriaForConditionalGeneration(config)
model = model.to(ms.bfloat16)

print("Loaded AriaForConditionalGeneration, time elapsed %.4fs" % (time.time() - start_time))

start_time = time.time()
processor = AriaProcessor.from_pretrained(model_id_or_path)
print("Loaded AriaProcessor, time elapsed %.4fs" % (time.time() - start_time))


image = None
# image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
image_path = "local_dir/cat.png"
if image_path.startswith("http"):
    image = Image.open(requests.get(image_path, stream=True).raw)
elif os.path.isfile(image_path):
    image = Image.open(image_path)

messages = [
    {
        "role": "user",
        "content": [
            {"text": None, "type": "image"},
            {"text": "what is the image?", "type": "text"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
print("input text:", [text])
# VQA: "<|im_start|>user\n<fim_prefix><|img|><fim_suffix>what is the image?<|im_end|>\n<|im_start|>assistant\n"
# text Q&A: "<|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant\n"
inputs = processor(text=text, images=image, return_tensors="np")

# convert input to Tensor
for key, value in inputs.items():
    inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    elif inputs[key].dtype != ms.bool_:
        inputs[key] = inputs[key].to(model.dtype)
    # "input_ids", "attention_mask", "pixel_values", "pixel_mask"
# print("inputs", inputs)

output = model.generate(
    **inputs,
    max_new_tokens=15,
    # stop_strings=["<|im_end|>"], # TODO: not support yet
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1] :]
result = processor.decode(output_ids, skip_special_tokens=True)

print(result)
