# 🎨 HunyuanImage-3.0: A Powerful Native Multimodal Model for Image Generation

This repository provides the inference codes of [HunyuanImage-3.0](https://arxiv.org/pdf/2509.23951), adapted from [official HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0) to support MindSpore.

-----

## ✨ Key Features

* 🧠 **Unified Multimodal Architecture:** Moving beyond the prevalent DiT-based architectures, HunyuanImage-3.0 employs a unified autoregressive framework. This design enables a more direct and integrated modeling of text and image modalities, leading to surprisingly effective and contextually rich image generation.

* 🏆 **The Largest Image Generation MoE Model:** This is the largest open-source image generation Mixture of Experts (MoE) model to date. It features 64 experts and a total of 80 billion parameters, with 13 billion activated per token, significantly enhancing its capacity and performance.

* 🎨 **Superior Image Generation Performance:** Through rigorous dataset curation and advanced reinforcement learning post-training, we've achieved an optimal balance between semantic accuracy and visual excellence. The model demonstrates exceptional prompt adherence while delivering photorealistic imagery with stunning aesthetic quality and fine-grained details.

* 💭 **Intelligent World-Knowledge Reasoning:** The unified multimodal architecture endows HunyuanImage-3.0 with powerful reasoning capabilities. It leverages its extensive world knowledge to intelligently interpret user intent, automatically elaborating on sparse prompts with contextually appropriate details to produce superior, more complete visual outputs.


## 📑 Todo List
- HunyuanImage-3.0 (Image Generation Model)
  - [x] Inference
  - [x] LoRA finetune
  - [ ] Image-to-Image Generation


## 📦 Requirements
| mindspore |	ascend driver | cann               |
| :-------: | :-----------: | :----------------: |
| >= 2.7.0  |  >= 25.2.0    | >= 8.2.RC1         |


## 🚀 Quick Start

### Installation
Clone the repo:
```sh
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/examples/hunyuanimage3_0
```

Install dependencies:
```sh
pip install -r requirements.txt
```

### Model Download

| Model                     | Params | Download | Recommended VRAM | Supported |
|---------------------------| --- | --- | --- | --- |
| HunyuanImage-3.0          | 80B total (13B active) | [HuggingFace](https://huggingface.co/tencent/HunyuanImage-3.0) | ≥ 3 × 80 GB | ✅ Text-to-Image
| HunyuanImage-3.0-Instruct | 80B total (13B active) | [HuggingFace](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct) | ≥ 3 × 80 GB | ✅ Text-to-Image<br>✅ Prompt Self-Rewrite <br>✅ CoT Think

Notes:
- Install performance extras (FlashAttention, FlashInfer) for faster inference.
- Multi‑NPU inference is recommended for the Base model.

Download Model Weights

```bash
# Download from HuggingFace
hf download tencent/HunyuanImage-3.0 --local-dir ./HunyuanImage-3
```

### Run HunyuanImage-3.0 Inference

```bash
sh infer_t2i.sh
```

#### Command Line Arguments in `run_image_gen.py`

| Arguments               | Description                                                  | Default     |
| ----------------------- | ------------------------------------------------------------ | ----------- |
| `--prompt`              | Input prompt                                                 | (Required)  |
| `--model-id`            | Model path                                                   | (Required)  |
| `--attn-impl`           | Attention implementation. Either `sdpa` or `flash_attention_2`. | `sdpa`      |
| `--moe-impl`            | MoE implementation. Either `eager` or `flashinfer`           | `eager`     |
| `--seed`                | Random seed for image generation                             | `None`      |
| `--diff-infer-steps`    | Diffusion infer steps                                        | `50`        |
| `--image-size`          | Image resolution. Can be `auto`, like `1280x768` or `16:9`   | `auto`      |
| `--save`                | Image save path.                                             | `image.png` |
| `--verbose`             | Verbose level. 0: No log; 1: log inference information.      | `0`         |
| `--rewrite`             | Whether to enable rewriting                                  | `1`         |
| `--sys-deepseek-prompt` | Select sys-prompt from `universal` or `text_rendering`       | `universal` |


### Run HunyuanImage-3.0 LoRA finetune

```bash
sh train_t2i.sh
```
