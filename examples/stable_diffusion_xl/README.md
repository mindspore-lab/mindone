# Stable Diffusion XL

This folder contains [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) models implemented with [MindSpore](https://www.mindspore.cn/), following the [official implementation](https://github.com/Stability-AI/generative-models) by Stability-AI.

## Features

> [!IMPORTANT]
>
> The features below only work on MindSpore 2.2.10~2.2.12 on Ascend 910*. The codebase follows the official implementation. We highly recommend you use [`mindone.diffusers`](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers) and [example/diffusers](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers) for more state-of-the-art SDXL training and inference tools.


### Inference

| Method            | [Model] SDXL-Base | [Model] SDXL-Pipeline | [Model] SDXL-Refiner | [Func] Samplers | [Func] Flash Attn |
|:-----------------:|:-----------------:|:---------------------:|:--------------------:|:---------------:|:-----------------:|
| Online Inference  | ✅                 | ✅                   | ✅                  | 7 samplers       | ✅                |
| Offline Inference | ✅                 | ❌                   | ❌                  | Euler EDM        | ✅                |


### Finetune with SDXL-1.0-Base

| Vallina | Dreambooth | LoRA |
|:-------:|:----------: |:----:|
| ✅       | ✅          | ✅    |

## Guide

1. Preparation
   - [Preparation](./docs/preparation.md)
   - [RankTable Generation](./tools/rank_table_generation/README.md)

2. Inference
    - [Online Infer](./docs/inference.md)
    - [Offline Infer](./offline_inference/README.md)

3. Finetune
    - [Config Guide](./docs/config_guide.md)
    - [Vanilla Finetune](./docs/vanilla_finetune.md)
    - [LoRA Finetune](./docs/lora_finetune.md)
    - [DreamBooth Finetune](./docs/dreambooth_finetune.md)

4. Others
   - [FAQ](./docs/faq_cn.md)
