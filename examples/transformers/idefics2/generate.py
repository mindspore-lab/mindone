import mindspore as ms
from mindone.transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor
from mindone.transformers import Idefics2ForConditionalGeneration
from mindone.transformers.image_utils import load_image

print("*"*50)
print("Loading Idefics2")
# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")

MODEL_HUB2 = "HuggingFaceM4/idefics2-8b"
processor = AutoProcessor.from_pretrained(MODEL_HUB2)
model = Idefics2ForConditionalGeneration.from_pretrained(
    MODEL_HUB2,
).set_train(False)

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ]
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

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
# ['User: What do we see in this image? \nAssistant: In this image, we can see the city of New York, and more specifically the Statue of Liberty. \nUser: And how about this image? \nAssistant: In this image we can see buildings, trees, lights, water and sky.']