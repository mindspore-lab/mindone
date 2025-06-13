from transformers import AutoProcessor

import mindspore as ms
from mindspore.nn import no_init_parameters

from mindone.transformers import SmolVLMForConditionalGeneration

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
with no_init_parameters():
    model = SmolVLMForConditionalGeneration.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", mindspore_dtype=ms.bfloat16
    )

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="np",
)
for k, v in inputs.items():
    inputs[k] = ms.tensor(v, dtype=ms.bfloat16 if k != "input_ids" else ms.int32)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
print(generated_texts)


# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://cdn.openai.com/sora/videos/tokyo-walk.mp4"},
            {"type": "text", "text": "Describe this video in detail"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="np",
)
for k, v in inputs.items():
    inputs[k] = ms.tensor(v, dtype=ms.bfloat16 if k != "input_ids" else ms.int32)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
