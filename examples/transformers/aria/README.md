# Aria: A Multimodal Vision-Language Model

[Aria](https://huggingface.co/rhymes-ai/Aria) is a multimodal vision-language model developed by Rhymes AI that can understand and generate responses based on both images and text. It supports various vision-language tasks including Visual Question Answering (VQA), image captioning, and multimodal chat.

## Introduction

Aria is designed to handle complex multimodal interactions, combining advanced vision understanding with natural language processing capabilities. The model can process images of various resolutions and generate coherent, contextually relevant responses.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |

## Quick Start

### Basic Usage

```python
import os
import time
import requests
from PIL import Image
from transformers import AriaProcessor
import mindspore as ms
from mindone.transformers import AriaForConditionalGeneration

ms.set_context(mode=ms.PYNATIVE_MODE)

# Load model and processor
model_id_or_path = "rhymes-ai/Aria"
processor = AriaProcessor.from_pretrained(model_id_or_path)

# For testing with smaller model
from transformers.models.aria import AriaConfig
import json
with open(os.path.join(model_id_or_path, "config.json"), "r") as json_file:
    config_json = json.load(json_file)
config_json["text_config"]["num_hidden_layers"] = 2
config_json["text_config"]["attn_implementation"] = "flash_attention_2"
config_json["vision_config"]["num_hidden_layers"] = 2
config_json["vision_config"]["attn_implementation"] = "eager"
config = AriaConfig(**config_json)
model = AriaForConditionalGeneration(config)
model = model.to(ms.bfloat16)

# Load image
image_path = "path/to/your/image.png"
if image_path.startswith("http"):
    image = Image.open(requests.get(image_path, stream=True).raw)
else:
    image = Image.open(image_path)

# Prepare input
messages = [
    {
        "role": "user",
        "content": [
            {"text": None, "type": "image"},
            {"text": "What is shown in this image?", "type": "text"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="np")

# Convert to MindSpore tensors
for key, value in inputs.items():
    inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    elif inputs[key].dtype != ms.bool_:
        inputs[key] = inputs[key].to(model.dtype)

# Generate response
output = model.generate(
    **inputs,
    max_new_tokens=50,
    tokenizer=processor.tokenizer,
    do_sample=True,
    temperature=0.9
)

output_ids = output[0][inputs["input_ids"].shape[1]:]
result = processor.decode(output_ids, skip_special_tokens=True)
print(result)
```

### Run the Example

```bash
# Make sure you have an image file at local_dir/cat.png or modify the path
python generate.py
```

## Features

- **Multimodal Understanding**: Processes both images and text inputs simultaneously
- **Flexible Image Input**: Supports various image formats and resolutions
- **Conversational Interface**: Uses chat-style message format for interactions
- **Customizable Generation**: Supports various sampling parameters and length controls

## Model Architecture

Aria combines:
- Vision encoder for image understanding
- Text encoder/decoder for language processing
- Multimodal fusion mechanisms for cross-modal interaction

## Performance

Aria demonstrates strong performance on various vision-language benchmarks including VQA, image captioning, and multimodal reasoning tasks.</contents>
</xai:function_call
<xai:function_call name="write">
<parameter name="file_path">/Users/weizheng/work/tmp/mindone/examples/transformers/bert/README.md
