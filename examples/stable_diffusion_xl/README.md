# Stable Diffusion XL

This folder contains [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) models implemented with [MindSpore](https://www.mindspore.cn/), reference to [Official Implementation](https://github.com/Stability-AI/generative-models) by Stability-AI.

## Features

- [x] Text-to-image generation with SDXL-1.0-Base.
- [ ] Image-to-image generation with SDXL-1.0-Refiner.
- [x] Support SoTA diffusion process samplers including EulerEDMSampler, etc. (under continuous update)
- [ ] Vanilla SDXL-1.0-Base fine-tune.
- [x] [LoRA](https://arxiv.org/abs/2106.09685) SDXL-1.0-Base fine-tune.
- [ ] Quantitative evaluation for diffusion models: FID, CLIP-Score
- [x] Memory Efficient Sampling and Tuning: [Flash-Attention](https://arxiv.org/abs/2205.14135), Auto-Mix-Precision, Recompute, etc. (under continuous update)

## What is New

- 2023/08/30

1. Support SDXL-1.0-Base models text-to-image generative.
2. Support SDXL-1.0-Base LoRA fine-tune.
3. Support Efficient Memory Sampling and Tuning.

## Dependency

- mindspore >= 2.0.1 or 2.1
- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run

```shell
pip install -r requirements.txt
```

## Getting Started

See [GETTING STARTED](GETTING_STARTED.md) for details.

## Notes

⚠️ This function is experimental. The script fine-tunes the whole model and often times the model overfits and runs into issues like catastrophic forgetting. It's recommended to try different hyper-parameters to get the best result on your dataset.
