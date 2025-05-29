import time

# from transformers import AutoProcessor
# from mindone.transformers import AutoModelForVision2Seq
from transformers import Idefics3Processor

import mindspore as ms

from mindone.transformers import Idefics3ForConditionalGeneration
from mindone.transformers.image_utils import load_image

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

MODEL_HUB = "HuggingFaceM4/Idefics3-8B-Llama3"
start_time = time.time()
processor = Idefics3Processor.from_pretrained(MODEL_HUB)
print("Loaded Idefics3Processor, time elapsed: %.4fs" % (time.time() - start_time))

start_time = time.time()
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_HUB, mindspore_dtype=ms.bfloat16, attn_implementation="eager"
)
print("Loaded Idefics3ForConditionalGeneration, time elapsed: %.4fs" % (time.time() - start_time))

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty.",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="np")
for k, v in inputs.items():
    inputs[k] = ms.tensor(v)
    if inputs[k].dtype == ms.int64:
        inputs[k] = inputs[k].to(ms.int32)
    else:
        inputs[k] = inputs[k].to(model.dtype)
print(inputs)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
