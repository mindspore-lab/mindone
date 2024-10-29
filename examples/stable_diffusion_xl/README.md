# Stable Diffusion XL

This folder contains [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) models implemented with [MindSpore](https://www.mindspore.cn/), following the [official implementation](https://github.com/Stability-AI/generative-models) by Stability-AI.

## Features

> [!IMPORTANT]
>
> The features works on MindSpore 2.3.1 on Ascend 910* and mainly follow the offical codebase. We highly recommend you use [`mindone.diffusers`](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers) and [example/diffusers](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers) for more state-of-the-art SDXL training and inference tools.


### Inference

| [Model] SDXL-Base | [Model] SDXL-Pipeline | [Model] SDXL-Refiner | [Func] Samplers | [Func] Flash Attn |
|:-----------------:|:---------------------:|:--------------------:|:---------------:|:-----------------:|
| ✅                 | ✅                   | ✅                  | 7 samplers       | ✅                |


### Finetune with SDXL-1.0-Base

| Vallina | LoRA |
|:-------:|:----:|
| ✅      | ✅   |

## Guide

1. Preparation
   - [Preparation](./docs/preparation.md)
   - [RankTable Generation](./tools/rank_table_generation/README.md)

2. Inference
    - [Inference](./docs/inference.md)

3. Finetune
    - [Config Guide](./docs/config_guide.md)
    - [Vanilla Finetune](./docs/vanilla_finetune.md)
    - [LoRA Finetune](./docs/lora_finetune.md)

4. Others
   - [FAQ](./docs/faq_cn.md)
