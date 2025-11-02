# Kimi-VL: Multimodal Vision-Language Model

[Kimi-VL](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) is an advanced multimodal vision-language model developed by Moonshot AI. It combines visual understanding with natural language processing to handle complex multimodal tasks including image description, visual question answering, and multimodal reasoning.

## Introduction

Kimi-VL is designed to excel at understanding and generating responses that involve both visual and textual information. The model supports various vision-language tasks and provides accurate, contextually aware responses for multimodal interactions.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |




## Quick Start

### Basic Multimodal Usage

```python
import os
import urllib.request
from kimi_vl import KimiVLConfig, KimiVLForConditionalGeneration
from PIL import Image
from transformers import AutoProcessor
import mindspore as ms
import mindspore.nn as nn

# Model configuration
DEBUG = False  # Set to True for debugging with smaller model
MODEL_PATH = "moonshotai/Kimi-VL-A3B-Instruct"

def main():
    if DEBUG:
        # Load smaller model for debugging
        ms.runtime.launch_blocking()
        config = KimiVLConfig.from_pretrained(MODEL_PATH, attn_implementation="flash_attention_2")
        config.text_config.num_hidden_layers = 2
        config.vision_config.num_hidden_layers = 1
        model = KimiVLForConditionalGeneration._from_config(config, torch_dtype=ms.bfloat16)
    else:
        # Load full model
        with nn.no_init_parameters():
            model = KimiVLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                mindspore_dtype=ms.bfloat16,
                attn_implementation="flash_attention_2"
            )

    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Prepare image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    image_path = "cat.png"
    urllib.request.urlretrieve(image_url, image_path)
    image = Image.open(image_path)

    # Prepare text input
    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": "Describe this image in detail.", "type": "text"}
            ]
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="np")

    # Convert to MindSpore tensors
    for key, value in inputs.items():
        tensor = ms.Tensor(value)
        if tensor.dtype == ms.int64:
            tensor = tensor.to(ms.int32)
        elif tensor.dtype != ms.bool_:
            tensor = tensor.to(model.dtype)
        inputs[key] = tensor

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Decode response
    response_ids = output[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(response_ids, skip_special_tokens=True)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
```

### Run the Example

```bash
python generate.py
```

## Model Variants

- **Kimi-VL-A3B-Instruct**: Instruction-tuned model for conversational tasks
- **Kimi-VL-A3B-Thinking**: Enhanced reasoning capabilities for complex tasks

## Features

- **Multimodal Understanding**: Processes images and text simultaneously
- **High-Resolution Support**: Handles various image resolutions effectively
- **Conversational Interface**: Supports chat-style interactions
- **Reasoning Capabilities**: Advanced reasoning for complex multimodal tasks
- **Flexible Input**: Accepts various image formats and text inputs

## Model Architecture

Kimi-VL combines:
- **Vision Encoder**: Advanced image understanding capabilities
- **Language Model**: Powerful text generation and understanding
- **Multimodal Fusion**: Seamless integration of visual and textual information
- **Attention Mechanisms**: Optimized attention for cross-modal interactions

## Performance

Kimi-VL demonstrates excellent performance on multimodal benchmarks:

- **Vision-Language Tasks**: State-of-the-art results on VQA and image captioning
- **Multimodal Reasoning**: Strong performance on complex reasoning tasks
- **Instruction Following**: Excellent alignment with user instructions
- **General Understanding**: Broad capabilities across various multimodal scenarios

## Use Cases

- **Visual Question Answering**: Answering questions about images
- **Image Captioning**: Generating detailed descriptions of images
- **Multimodal Chat**: Conversational AI with image understanding
- **Document Analysis**: Understanding and describing document content
- **Educational Applications**: Visual learning and explanation
- **Content Moderation**: Analyzing images and text for appropriateness
- **Accessibility**: Helping visually impaired users understand visual content</contents>
</xai:function_call
<xai:function_call name="write">
<parameter name="file_path">/Users/weizheng/work/tmp/mindone/examples/transformers/qwen/README.md
