# Helium: Efficient Language Model

[Helium](https://huggingface.co/kyutai/helium-1-preview-2b) is an efficient language model developed by Kyutai that provides fast and accurate text generation capabilities. It comes in different sizes (2B and 7B parameters) and is optimized for various natural language processing tasks.

## Introduction

Helium is designed to be an efficient alternative to larger language models while maintaining high performance. It uses advanced architectural optimizations to provide fast inference speeds while delivering coherent and contextually relevant text generation.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |




## Quick Start

### Text Generation

```python
from time import time
from transformers import AutoTokenizer
import mindspore as ms
from mindspore import Tensor
from mindone.transformers import HeliumForCausalLM

def main():
    # Choose model size
    model_id = "kyutai/helium-1-preview-2b"  # or "google/helium-7b"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = HeliumForCausalLM.from_pretrained(model_id, mindspore_dtype=ms.float16)

    # Prepare input
    prompt = "What is your favorite condiment?"
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]

    # Ensure proper shape
    input_ids = (
        Tensor(input_ids) if (len(input_ids.shape) == 2 and input_ids.shape[0] == 1)
        else Tensor(input_ids).unsqueeze(0)
    )

    # Generate text
    infer_start = time()
    generate_ids = model.generate(input_ids, max_length=30)
    print(f"Inference time: {time() - infer_start:.3f}s")

    # Decode and print result
    result = tokenizer.batch_decode(
        generate_ids[:, input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")

if __name__ == "__main__":
    main()
```

### Run the Example

```bash
python generate.py
```

## Model Variants

- **Helium-2B**: Lightweight model with 2 billion parameters, optimized for speed
- **Helium-7B**: Larger model with 7 billion parameters, better performance for complex tasks

## Features

- **Efficient Architecture**: Optimized for fast inference and low memory usage
- **High Performance**: Maintains strong performance despite smaller size
- **Flexible Deployment**: Suitable for various hardware configurations
- **Easy Integration**: Compatible with standard transformers interface

## Model Architecture

Helium uses advanced transformer architecture with optimizations for:
- **Reduced Parameter Count**: Efficient parameter utilization
- **Fast Attention Mechanisms**: Optimized attention computation
- **Memory Efficiency**: Lower memory footprint during inference

## Performance

Helium models provide excellent performance-to-efficiency ratios:

- **Speed**: Fast inference times suitable for real-time applications
- **Quality**: High-quality text generation comparable to larger models
- **Efficiency**: Lower computational requirements than similar-sized models
- **Scalability**: Good performance scaling across different hardware

## Use Cases

- **Content Generation**: Blog posts, articles, creative writing
- **Chatbots**: Conversational AI applications
- **Code Assistance**: Programming help and code generation
- **Summarization**: Text summarization tasks
- **Question Answering**: Answering questions based on context
- **Language Translation**: Efficient translation tasks</contents>
</xai:function_call
<xai:function_call name="write">
<parameter name="file_path">/Users/weizheng/work/tmp/mindone/examples/transformers/herbert/README.md
