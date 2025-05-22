import time

from transformers import AriaProcessor

import mindspore as ms

from mindone.transformers import AriaForConditionalGeneration

# from functools import partial
# import mindspore.mint.distributed as dist
# from mindspore.communication import GlobalComm
# from mindone.trainers.zero import prepare_network

ms.set_context(mode=ms.PYNATIVE_MODE)
# dist.init_process_group(backend="hccl")
# ms.set_auto_parallel_context(parallel_mode="data_parallel")


# model_id_or_path = "rhymes-ai/Aria"
model_id_or_path = "/home/susan/workspace/checkpoints/Aria"

start_time = time.time()
import json

# DEBUG
from transformers.models.aria import AriaConfig

with open("/home/susan/workspace/checkpoints/Aria/config.json", "r") as json_file:
    config_json = json.load(json_file)
config_json["text_config"]["num_hidden_layers"] = 2  # testing use
config = AriaConfig(**config_json)
model = AriaForConditionalGeneration(config)
model = model.to(ms.bfloat16)
# DEBUG
# model = AriaForConditionalGeneration.from_pretrained(
#     model_id_or_path, mindspore_dtype=ms.bfloat16
# )
# shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
# model = shard_fn(model)
print("Loaded AriaForConditionalGeneration, time elapsed %.4fs" % (time.time() - start_time))

start_time = time.time()
processor = AriaProcessor.from_pretrained(model_id_or_path)
print("Loaded AriaProcessor, time elapsed %.4fs" % (time.time() - start_time))

# DEBUG

# WAITING for idefics3
image = None
# uncomment below with vision tower
# image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
# image = Image.open(requests.get(image_path, stream=True).raw)
messages = [
    {
        "role": "user",
        "content": [
            # {"text": None, "type": "image"},
            # {"text": "what is the image?", "type": "text"},
            {"text": "who are you?", "type": "text"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
print("input text:", text)
# "<|im_start|>user\nwho are you?<|im_end|>\n<|im_start|>assistant"
inputs = processor(text=text, images=image, return_tensors="np")

# convert input to Tensor
for key, value in inputs.items():
    inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    # "input_ids", "attention_mask", "image_grid_thw", "pixel_values"
print("inputs", inputs)

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
