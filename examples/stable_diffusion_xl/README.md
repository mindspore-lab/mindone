# Stable Diffusion XL

This folder contains [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) models implemented with [MindSpore](https://www.mindspore.cn/), reference to [Official Implementation](https://github.com/Stability-AI/generative-models) by Stability-AI.

## Features

- [x] Infer: Text-to-image generation with SDXL-1.0-Base/SDXL-1.0-PipeLine.
- [x] Infer: Image-to-image generation with SDXL-1.0-Refiner.
- [x] Infer: Support 7 SoTA diffusion process samplers.
- [x] Infer: Support with MSLite.
- [x] Infer: Support T2I-Adapters for Text-to-Image generation with extra visual guidance.
- [x] Infer: Support [ControlNet](https://arxiv.org/abs/2302.05543) inference with SDXL-1.0-Base.
- [x] (⚠️experimental) Finetune: [LoRA](https://arxiv.org/abs/2106.09685) fine-tune with SDXL-1.0-Base.
- [x] (⚠️experimental) Finetune: [DreamBooth](https://arxiv.org/abs/2208.12242) lora fine-tune with SDXL-1.0-Base.
- [x] (⚠️experimental) Finetune: [Textual Inversion](https://arxiv.org/abs/2208.01618) fine-tune with SDXL-1.0-Base.
- [x] (⚠️experimental) Finetune: Vanilla fine-tune with SDXL-1.0-Base.
- [x] LoRA model conversion for Torch inference, refer to [tutorial](tools/lora_conversion/README_CN.md)
- [x] Memory Efficient Sampling and Tuning: [Flash-Attention](https://arxiv.org/abs/2205.14135), Auto-Mix-Precision, Recompute, etc. (under continuous update)

## Documentation

1. Preparation
   - [Installation](./installation.md)
   - [Weight Convert](./weight_convertion.md)
   - [RankTable Generation](./tools/rank_table_generation/README.md)
2. Inference
    - [Online Infer](./inference.md)
    - [Offline Infer](./offline_inference/README.md)
    - [LCM Infer](./inference_lcm.md)
3. Finetune
    - [Vanilla Finetune](./vanilla_finetune.md)
    - [LoRA Finetune](./GETTING_STARTED.md)
    - [DreamBooth Finetune](dreambooth_finetune.md)
    - [Textual Inversion Finetune](textual_inversion_finetune.md)

## What is New

**Jan 30, 2024**
1. Add [ControlNet](controlnet.md) inference support for SDXL

**Jan 18, 2024**
1. Support latent/text-embedding cache.
2. Support vanilla fine-tune with PerBatchSize of 6 on 910*.

**Jan 10, 2024**
1. Support [Textual Inversion](https://arxiv.org/abs/2208.01618) fine-tune.

**Nov 22, 2023**
1. Support [DreamBooth](https://arxiv.org/abs/2208.12242) lora fine-tune.
2. Support [Offline Infer](./offline_inference/README.md) with MSLite.
3. Support Vanilla fine-tune.
4. Adapted to [MindSpore 2.2](https://www.mindspore.cn/install).
5. Add [T2I-Adapters](../t2i_adapter/README.md) support for SDXL.

**Sep 15, 2023**
1. Support SDXL-1.0-Refiner model for image-to-image generation.
2. Support SDXL-1.0-Refiner [LoRA](https://arxiv.org/abs/2106.09685) fine-tune.
3. Support SDXL-1.0-PipeLine for txt-to-image generation.
4. Support Multi-Aspect data process (e.g., [sample config](./configs/training/sd_xl_base_finetune_multi_aspect.yaml)).
5. Improve sampling speed (e.g., [sdxl-base](./configs/inference/sd_xl_base.yaml) sampled 40 steps on Ascend910A: 125s -> 21s).
6. Adapted to [MindSpore 2.1.0](https://www.mindspore.cn/install).

**Aug 30, 2023**
1. Support SDXL-1.0-Base model for text-to-image generation.
2. Support SDXL-1.0-Base [LoRA](https://arxiv.org/abs/2106.09685) fine-tune.
3. Support Efficient Memory Sampling and Tuning.

## Getting Started

See [GETTING STARTED](GETTING_STARTED.md) for details.

## Examples:

Note: sampled 40 steps by SDXL-1.0-Base on Ascend 910 (online infer).

<div align="center">
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/68d132e1-a954-418d-8cb8-5be4d8162342" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/9f0d0d2a-2ff5-4c9b-a0d0-1c744762ee92" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/dbaf0c77-d8d3-4457-b03c-82c3e4c1ba1d" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/f52168ef-53aa-4ee9-9f17-6889f10e0afb" width="240" />
</div>
<p align="center">
<font size=3>
<em> Fig1: "Vibrant portrait painting of Salvador Dalí with a robotic half face." </em> <br>
<em> Fig2: "A capybara made of voxels sitting in a field." </em> <br>
<em> Fig3: "Cute adorable little goat, unreal engine, cozy interior lighting, art station, detailed’ digital painting, cinematic, octane rendering." </em> <br>
<em> Fig4: "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest that says "SDXL"!." </em> <br>
</font>
</p>
<br>
