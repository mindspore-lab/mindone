import time

from PIL import Image
from transformers import GotOcr2ImageProcessor, InternVLProcessor

import mindspore as ms

from mindone.transformers import InternVLForConditionalGeneration

MODEL_HUB = "OpenGVLab/InternVL3-1B-hf"
image = "demo.jpeg"

# Load processor
start = time.time()
processor = InternVLProcessor.from_pretrained(MODEL_HUB)
# GotOcr2ImageProcessorFast does not support return_tensors="np", use GotOcr2ImageProcessor instead
image_processor = GotOcr2ImageProcessor.from_pretrained(MODEL_HUB)
processor.image_processor = image_processor
print(f"Loaded InternVLProcessor in {time.time()-start:.4f}s")

# Load model with bfloat16 and eager attention
start = time.time()
model = InternVLForConditionalGeneration.from_pretrained(
    MODEL_HUB,
    mindspore_dtype=ms.bfloat16,
    attn_implementation="eager",
)
print(f"Loaded model in {time.time()-start:.4f}s")

# load image
image = Image.open(image)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# Tokenize + encode
inputs = processor(text=prompt, images=[image], return_tensors="np")

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
