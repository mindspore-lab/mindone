import time
import mindspore as ms

from transformers import InternVLProcessor
from mindone.transformers import InternVLForConditionalGeneration
from mindone.transformers.image_utils import load_image

# Load images (or directly use PIL.Image.open() if preferred)
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")

MODEL_HUB = "OpenGVLab/InternVL3-1B"

# Load processor
start = time.time()
processor = InternVLProcessor.from_pretrained(MODEL_HUB)
print(f"Loaded InternVLProcessor in {time.time()-start:.4f}s")

# Load model with bfloat16 and eager attention
start = time.time()
model = InternVLForConditionalGeneration.from_pretrained(
    MODEL_HUB,
    mindspore_dtype=ms.bfloat16,
    attn_implementation="eager",
)
print(f"Loaded model in {time.time()-start:.4f}s")

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

# Tokenize + encode
inputs = processor(text=prompt, images=[image1, image2], return_tensors="np")

for k, v in inputs.items():
    tensor = ms.Tensor(v)
    if tensor.dtype == ms.int64:
        tensor = tensor.astype(ms.int32)
    else:
        tensor = tensor.astype(model.dtype)
    inputs[k] = tensor

# Generate
start = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=500)
print(f"Inference in {time.time()-start:.4f}s")

# Decode
texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(texts)
