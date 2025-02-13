# Qwen2-VL
Qwen2-VL is a multimodal vision-language model series based on Qwen2, which supports inputs of text, arbitrary-resolution image, long video (20min+) and multiple languages. 

# Quick Start

Tested requirements:
- python==3.9.19
- mindspore==2.3.1
- transformers=4.45.2
- tokenizers==0.20.0

Tested retrained weights from huggingface hub:
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- 

```python
from transformers import AutoTokenizer, AutoProcessor
from mindone.transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info
from mindspore import Tensor

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen2/Qwen2-VL-7B-Instruct", mindspore_dtype=ms.float32)
processor = AutoProcessor.from_pretrained("Qwen2/Qwen2-VL-7B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "demo.jpeg",
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
generated_ids = model.generate(Tensor(inputs.input_ids, dtype=ms.int32), max_new_tokens=128)
generated_ids_trimmed = 
    [out_ids[len(in_ids) :] for in_ids, out_ids 
    in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
```

# Tutorial of Qwen2-VL
[Qwen2-VL Implementation Tutorial (MindSpore Version)](tutorial.md)