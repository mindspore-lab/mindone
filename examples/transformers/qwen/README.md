# Qwen2: Advanced Language Model

[Qwen2](https://huggingface.co/Qwen) is a series of large language models developed by Alibaba Cloud, featuring strong performance across various natural language processing tasks. Qwen2 models are designed to be efficient, versatile, and capable of handling complex language understanding and generation tasks.

## Introduction

Qwen2 represents the next generation of Qwen models with improved architecture, training data, and capabilities. The models support multiple languages and excel at various NLP tasks including text generation, code generation, mathematical reasoning, and general language understanding.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |




## Quick Start

### Interactive Text Generation

```python
import argparse
import ast
import os
import time
from transformers import AutoTokenizer
import mindspore as ms
from mindone.transformers.mindspore_adapter import auto_mixed_precision
from mindone.transformers.models.qwen2 import Qwen2ForCausalLM

def run_qwen2_generate(args):
    print("=====> Building model...")

    s_time = time.time()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = Qwen2ForCausalLM.from_pretrained(args.model_path, use_flash_attention_2=args.use_fa)

    # Apply mixed precision
    model = auto_mixed_precision(model, amp_level="O2", dtype=ms.float16)

    print("=====> Building model done.")
    print(f"Model loading time: {time.time() - s_time:.2f}s")

    # Interactive generation loop
    while True:
        prompt = input("Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: ")

        if prompt == "q":
            break

        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
        input_tensor = ms.Tensor(input_ids)

        # Generate response
        print("Generating response...")
        start_time = time.time()

        output = model.generate(
            input_tensor,
            max_length=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generation_time = time.time() - start_time
        print(".2f")

        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-7B",
                       help="Path to pretrained model")
    parser.add_argument("--use_fa", action="store_true",
                       help="Use flash attention for faster inference")
    args = parser.parse_args()

    run_qwen2_generate(args)
```

### Run the Example

```bash
# Run interactive Qwen2 text generation
python generate.py --model_path Qwen/Qwen2-7B

# Run with flash attention for better performance
python generate.py --model_path Qwen/Qwen2-7B --use_fa
```

## Model Variants

- **Qwen2-0.5B**: Lightweight model for basic tasks
- **Qwen2-1.5B**: Small model with good performance
- **Qwen2-7B**: Standard model balancing performance and efficiency
- **Qwen2-72B**: Large model for complex tasks

## Features

- **Multilingual Support**: Supports multiple languages including English, Chinese, etc.
- **Code Generation**: Strong capabilities in programming and code-related tasks
- **Mathematical Reasoning**: Excellent performance on mathematical problems
- **Long Context**: Supports extended context windows
- **Efficient Architecture**: Optimized transformer architecture for better performance

## Model Architecture

Qwen2 uses advanced transformer architecture with:
- **Grouped Query Attention**: Improved attention mechanism for efficiency
- **Rotary Position Embedding (RoPE)**: Better positional encoding
- **SwiGLU Activation**: Enhanced activation functions
- **Pre-normalization**: Improved training stability

## Performance

Qwen2 achieves excellent results on various benchmarks:

- **MMLU**: Outstanding performance on academic benchmarks
- **GSM8K**: Strong mathematical reasoning capabilities
- **HumanEval**: Excellent code generation performance
- **Multilingual Tasks**: Superior performance across languages
- **Long Context Tasks**: Effective handling of extended contexts

## Use Cases

- **Conversational AI**: Building chatbots and virtual assistants
- **Content Generation**: Creating articles, stories, and marketing content
- **Code Assistance**: Programming help and code generation
- **Educational Applications**: Tutoring and learning assistance
- **Research**: Academic research and analysis
- **Business Applications**: Customer service, data analysis, automation
- **Creative Writing**: Poetry, scripts, and creative content generation</contents>
</xai:function_call
<xai:function_call name="write">
<parameter name="file_path">/Users/weizheng/work/tmp/mindone/examples/transformers/qwen2_audio/README.md
