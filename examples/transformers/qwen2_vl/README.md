# Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
[Paper](https://arxiv.org/abs/2409.12191) | [HF Model Card](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

> **Qwen2-VL** is a multimodal vision-language model series based on Qwen2, which supports inputs of text, arbitrary-resolution image, long video (20min+) and multiple languages.

# Get Started

## Requirements:
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.4.1 | 24.1RC3 | 7.3.0.1.231 | 8.0.RC2.beta1|

Tested with:
- python==3.10.16
- mindspore==2.4.1
- transformers=4.46.3
- tokenizers==0.20.0
- mindone

Pretrained weights from huggingface hub: [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

## Quick Start


`test_vqa.py` and `video_understanding.py` provides examples of image and video VQA. Here is an usage example of image understanding:

```python
from transformers import AutoProcessor
from mindone.transformers import Qwen2VLForConditionalGeneration
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info
from mindspore import Tensor
import numpy as np

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen2/Qwen2-VL-7B-Instruct", mindspore_dtype=ms.float32)
processor = AutoProcessor.from_pretrained("Qwen2/Qwen2-VL-7B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "demo.jpeg", # REPLACE with your own image
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="np",
)
# convert input to Tensor
for key, value in inputs.items():
    if isinstance(value, np.ndarray):
        inputs[key] = ms.Tensor(value)
    elif isinstance(value, list):
        inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]
print(output_text)
```

# Tutorial of Qwen2-VL
[Qwen2-VL Implementation Tutorial (MindSpore Version)](tutorial.md)
