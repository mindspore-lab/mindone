# Stable Diffusion XL

This folder contains [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) models implemented with [MindSpore](https://www.mindspore.cn/), following the [official implementation](https://github.com/Stability-AI/generative-models) by Stability-AI.

> [!IMPORTANT]
>
> The features works on MindSpore 2.3.1 on Ascend 910* and mainly follow the offical codebase. We highly recommend you use [`mindone.diffusers`](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers) and [example/diffusers](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers) for more state-of-the-art SDXL training and inference tools.


## Guide

- Preparation: [Requirements & Weight Conversions & Datasets Preparetion](./docs/preparation.md)

- Inference: [Inference Tutorials & Performance](./docs/inference.md)

- Finetune: [Config Guide](./docs/config_guide.md) & [Vanilla Finetune](./docs/vanilla_finetune.md) & [LoRA Finetune](./docs/lora_finetune.md)

- Others: [Frequently Asked Questions](./docs/faq_cn.md)
