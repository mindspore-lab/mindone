# Stable Diffusion XL

This folder contains [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) models implemented with [MindSpore](https://www.mindspore.cn/), reference to [Official Implementation](https://github.com/Stability-AI/generative-models) by Stability-AI.

## Features

- [x] Text-to-image generation with SDXL-1.0-Base/SDXL-1.0-PipeLine(Base+Refiner).
- [x] Image-to-image generation with SDXL-1.0-Refiner.
- [x] Support 7 SoTA diffusion process samplers.
- [x] [LoRA](https://arxiv.org/abs/2106.09685) fine-tune with SDXL-1.0-Base/SDXL-1.0-Refiner.
- [ ] [DreamBooth](https://arxiv.org/abs/2208.12242) fine-tune with SDXL-1.0-Base.
- [ ] Vanilla fine-tune.
- [ ] Quantitative evaluation for diffusion models: FID, CLIP-Score
- [x] Memory Efficient Sampling and Tuning: [Flash-Attention](https://arxiv.org/abs/2205.14135), Auto-Mix-Precision, Recompute, etc. (under continuous update)

## What is New

**Sep 15, 2023**

1. Support SDXL-1.0-Refiner model for image-to-image generation.
2. Support SDXL-1.0-Refiner [LoRA](https://arxiv.org/abs/2106.09685) fine-tune.
3. Support SDXL-1.0-PipeLine for txt-to-image generation.
4. Support Multi-Aspect data process (e.g., [sample config](./configs/training/sd_xl_base_finetune_multi_aspect.yaml)).
5. Faster sampling speed (e.g., [sdxl-base](./configs/inference/sd_xl_base.yaml) sampled 40 steps on Ascend910A: 125s -> 21s).
6. Adapted to [MindSpore 2.1.0](https://www.mindspore.cn/install).

**Aug 30, 2023**

1. Support SDXL-1.0-Base model for text-to-image generation.
2. Support SDXL-1.0-Base [LoRA](https://arxiv.org/abs/2106.09685) fine-tune.
3. Support Efficient Memory Sampling and Tuning.

## Getting Started

See [GETTING STARTED](GETTING_STARTED.md) for details.

## Examples:

Note: sampled 40 steps by SDXL-1.0-Base on Ascend 910A.

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
