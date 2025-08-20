import time
from transformers import InternVLForConditionalGeneration
from transformers import InternVLProcessor
from PIL import Image

MODEL_HUB = "OpenGVLab/InternVL3-1B-hf"
image = "demo.jpeg"

start = time.time()
processor = InternVLProcessor.from_pretrained("model/InternVL3-1B")
print(f"Loaded InternVLProcessor in {time.time()-start:.4f}s")

print("loading the model...")
start = time.time()
model = InternVLForConditionalGeneration.from_pretrained(
    MODEL_HUB,
    attn_implementation="eager",
)
print(f"Loaded model in {time.time()-start:.4f}s")

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

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
)

start = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=500)
print(f"Inference in {time.time()-start:.4f}s")

# decode
texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(texts)