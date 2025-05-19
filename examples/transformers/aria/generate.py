import requests
import mindspore as ms
from PIL import Image
from mindone.transformers import AutoModelForCausalLM
from transformers import AutoProcessor

ms.set_context(mode=ms.PYNATIVE_MODE)

model_id_or_path = "rhymes-ai/Aria"

model = AutoModelForCausalLM.from_pretrained(model_id_or_path, mindspore_dtype=ms.bfloat16)  # trust_remote_code=True

processor = AutoProcessor.from_pretrained(model_id_or_path) # trust_remote_code=True

image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"

image = Image.open(requests.get(image_path, stream=True).raw)

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
inputs = processor(text=text, images=image, return_tensors="np")

# convert input to Tensor
for key, value in inputs.items():
    inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    # "input_ids", "attention_mask", "image_grid_thw", "pixel_values"
print("inputs", inputs)
# inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

output = model.generate(
    **inputs,
    max_new_tokens=500,
    stop_strings=["<|im_end|>"],
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9,
)
output_ids = output[0][inputs["input_ids"].shape[1]:]
result = processor.decode(output_ids, skip_special_tokens=True)

print(result)