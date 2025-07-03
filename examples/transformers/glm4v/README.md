# GLM-4.1V

## Introduction
Vision-Language Models (VLMs) have become foundational components of intelligent systems. As real-world AI tasks grow increasingly complex, VLMs must evolve beyond basic multimodal perception to enhance their reasoning capabilities in complex tasks. This involves improving accuracy, comprehensiveness, and intelligence, enabling applications such as complex problem solving, long-context understanding, and multimodal agents.

Based on the GLM-4-9B-0414 foundation model, we present the new open-source VLM model GLM-4.1V-9B-Thinking, designed to explore the upper limits of reasoning in vision-language models. By introducing a "thinking paradigm" and leveraging reinforcement learning, the model significantly enhances its capabilities. It achieves state-of-the-art performance among 10B-parameter VLMs, matching or even surpassing the 72B-parameter Qwen-2.5-VL-72B on 18 benchmark tasks. We are also open-sourcing the base model GLM-4.1V-9B-Base to support further research into the boundaries of VLM capabilities.

Compared to the previous generation models CogVLM2 and the GLM-4V series, GLM-4.1V-Thinking offers the following improvements:

The first reasoning-focused model in the series, achieving world-leading performance not only in mathematics but also across various sub-domains.
Supports 64k context length.
Handles arbitrary aspect ratios and up to 4K image resolution.
Provides an open-source version supporting both Chinese and English bilingual usage.

# Get Started

## Requirements:
| mindspore   | 	ascend driver | firmware       | cann tookit/kernel|
|-------------|----------------|----------------| --- |
| 2.5.0/2.6.0 | 24.1.RC3.b080  | 7.5.T11.0.B088 | 8.0.RC3.beta1|

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
cd examples/transformers/glm4v
pip install -r requirements.txt
```

## Quick Start

Here is a usage example of GLM-4.1V-Thinking. you can use the following command:

```bash
# for GLM-4.1V-Thinking inference
python generate.py
    --model_name "THUDM/GLM-4.1V-9B-Thinking"
    --image "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    --prompt "Describe this image."
```

`model_name` and `image` could be replaced with your local path. Give it a try with various images and prompts🤗🤗.

## Inference Speed
|      model name	      | mindspore version |   precision*   | cards | flash attn | 	tokens/s	 |
|:---------------------:|:-----------------:|:--------------:|:---:  |:----------:|:----------:|
| THUDM/GLM-4.1V-9B-Thinking |       2.5.0       |   bf16 | 1 |     ✅      |    1.38    |
| THUDM/GLM-4.1V-9B-Thinking |       2.6.0       |   bf16 | 1 |     ✅      |    1.63    |
