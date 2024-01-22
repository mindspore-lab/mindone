# Stable Diffusion XL

这个文件夹包含使用 [MindSpore](https://www.mindspore.cn/) 实现的 [Stable Diffusion XL (SDXL)](https://arxiv.org/abs/2307.01952) 模型，参考自Stability-AI[官方实现](https://github.com/Stability-AI/generative-models)。
## Features

- [x] **推理：** 使用 SDXL-1.0-Base/SDXL-1.0-PipeLine 进行文本到图像生成。
- [x] **推理：** 使用 SDXL-1.0-Refiner 进行图像到图像生成。
- [x] **推理：** 支持7种SoTA扩散过程采样器。
- [x] **推理：** 支持MSLite。
- [x] **推理：** 支持T2I适配器，用于带有额外视觉引导的文本到图像生成。
- [x] **（⚠️实验性）微调：** 使用SDXL-1.0-Base进行[LoRA](https://arxiv.org/abs/2106.09685)微调。
- [x] **（⚠️实验性）微调：** 使用SDXL-1.0-Base进行[DreamBooth](https://arxiv.org/abs/2208.12242) lora微调。
- [x] **（⚠️实验性）微调：** 使用SDXL-1.0-Base进行[Textual Inversion](https://arxiv.org/abs/2208.01618)微调。
- [x] **（⚠️实验性）微调：** 使用SDXL-1.0-Base进行Vanilla微调。
- [x] **LoRA模型转换：** 用于Torch推理，参见[tutorial](tools/lora_conversion/README_CN.md)。
- [x] **内存高效采样和调整：** [Flash-Attention](https://arxiv.org/abs/2205.14135)，Auto-Mix-Precision，Recompute等（持续更新中）。

## 文档

1. 推理
    - [在线推理](./GETTING_STARTED.md)
    - [离线推理](./offline_inference/README.md)
2. 微调
    - [基础微调](./GETTING_STARTED.md)
    - [LoRA微调](./GETTING_STARTED.md)
    - [DreamBooth微调](dreambooth_finetune.md)
    - [Textual Inversion微调](textual_inversion_finetune.md)

## 更新内容
**2024年1月10日**
1. 支持[Textual Inversion](https://arxiv.org/abs/2208.01618)微调。

**2023年11月22日**

1. 支持[DreamBooth](https://arxiv.org/abs/2208.12242) lora微调。
2. 支持使用MSLite进行[离线推理](./offline_inference/README.md)。
3. 支持基础微调。
4. 适配[MindSpore 2.2](https://www.mindspore.cn/install)。
5. 为SDXL添加了[T2I适配器](../t2i_adapter/README.md)支持。

**2023年9月15日**

1. 支持SDXL-1.0-Refiner模型，用于图像生成。
2. 支持SDXL-1.0-Refiner的[LoRA](https://arxiv.org/abs/2106.09685)微调。
3. 支持SDXL-1.0-PipeLine，用于文本到图像生成。
4. 支持多方面数据处理（例如，[样本配置](./configs/training/sd_xl_base_finetune_multi_aspect.yaml)）。
5. 提高采样速度（例如，在Ascend910A上对[sdxl-base](./configs/inference/sd_xl_base.yaml)进行40个步骤的采样：125秒 -> 21秒）。
6. 适配[MindSpore 2.1.0](https://www.mindspore.cn/install)。

**2023年8月30日**

1. 支持SDXL-1.0-Base模型，用于文本到图像生成。
2. 支持SDXL-1.0-Base的[LoRA](https://arxiv.org/abs/2106.09685)微调。
3. 支持高效内存采样和调整。

## 入门指南

详见[入门指南](GETTING_STARTED.md)获取详情。


## 示例:

注意：使用SDXL-1.0-Base在Ascend 910A上采样了40步（在线推理）。

<div align="center">
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/68d132e1-a954-418d-8cb8-5be4d8162342" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/9f0d0d2a-2ff5-4c9b-a0d0-1c744762ee92" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/dbaf0c77-d8d3-4457-b03c-82c3e4c1ba1d" width="240" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/f52168ef-53aa-4ee9-9f17-6889f10e0afb" width="240" />
</div>
<p align="center">
<font size=3>
<em> 图1: "Vibrant portrait painting of Salvador Dalí with a robotic half face." </em> <br>
<em> 图2: "A capybara made of voxels sitting in a field." </em> <br>
<em> 图3: "Cute adorable little goat, unreal engine, cozy interior lighting, art station, detailed’ digital painting, cinematic, octane rendering." </em> <br>
<em> 图4: "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest that says "SDXL"!." </em> <br>
</font>
</p>
<br>
