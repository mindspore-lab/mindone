# Stable Diffusion XL

This folder contains [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) models implemented with [MindSpore](https://www.mindspore.cn/), with a reference to the [Official Implementation](https://github.com/Stability-AI/generative-models) by Stability-AI.

## Features

> [!IMPORTANT]
>
> All the features only work on MindSpore 2.2.10~2.2.12 on Ascend 910*. We do not plan on maintaining the features on the later MindSpore version. Instead, [`mindone.diffusers`](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers) and [example/diffusers](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers) are recommended for SDXL training and inference.


### Inference

| Method            | [Model] SDXL-Base | [Model] SDXL-Pipeline | [Model] SDXL-Refiner | [Func] Samplers | [Func] Flash Attn |
|:-----------------:|:-----------------:|:---------------------:|:--------------------:|:---------------:|:-----------------:|
| Online Inference  | ✅                 | ✅                   | ✅                  | 7 samplers       | ✅                |
| Offline Inference | ✅                 | ❌                   | ❌                  | Euler EDM        | ✅                |


### Finetune with SDXL-1.0-Base

| Vallina | Dreambooth | LoRA | Textual Inversion | ControlNet |
|:-------:|:---------- |:----:|:-----------------:|:----------:|
| ✅       | ✅          | ✅    | ✅                 | ✅          |

## Guide

1. Preparation
   - [Preparation](./docs/preparation.md)
   - [RankTable Generation](./tools/rank_table_generation/README.md)
   - [FAQ](./docs/faq_cn.md)

2. Inference
    - [Online Infer](./docs/inference.md)
    - [Offline Infer](./offline_inference/README.md)

3. Finetune

    * [Configuration Guidance](./docs/configuration_guidance.md)

    - [Vanilla Finetune](./docs/vanilla_finetune.md)
    - [LoRA Finetune](./docs/lora_finetune.md)
    - [DreamBooth Finetune](./docs/dreambooth_finetune.md)
    - [Textual Inversion Finetune](./docs/textual_inversion_finetune.md)
    - [ControlNet](./docs/controlnet.md)
