from transformers import AutoModelForVision2Seq, AutoProcessor

import mindspore as ms

model_path = "ibm-granite/granite-vision-3.1-2b-preview"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path)

img_path = "ibm-granite/granite-vision-3.1-2b-preview/example.png"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": img_path},
            {"type": "text", "text": "What is the highest scoring model on ChartQA and what is its score?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
)
inputs = {k: ms.Tensor(v) for k, v in inputs.items()}

output = model.generate(**inputs, max_new_tokens=100)
print(
    processor.decode(output[0], skip_special_tokens=True)
)  # The highest scoring model on ChartQA is Granite Vision 3.1 with a score of 0.86
