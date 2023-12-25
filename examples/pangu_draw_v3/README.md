[中文](./README_CN.md)|[English](./README.md)

# PanGu Draw 3.0

This folder contains **PanGu Draw 3.0** models implemented with MindSpore.


## Features

In contrast to version 2.0, Pangu Draw 3.0 has been subject to experimentation and updates across various aspects, including multi-language support, diverse resolutions, improved image quality, and model scaling. This includes:

- [x] The current industry's largest 5-billion-parameter Chinese text-to-image model.
- [x] Supports bilingual input in both Chinese and English.
- [x] Supports output of native 1K resolution images.
- [x] Outputs images in multiple size ratios.
- [x] Quantifiable stylized adjustments: cartoon, aesthetic, photography controller.
- [x] Based on Ascend+MindSpore for large-scale training and inference, using a self-developed MindSpore platform and Ascend 910 hardware.
- [x] Utilizes self-developed RLAIF to enhance image quality and artistic expression.


## What is New

**Dec 12, 2023**

Support inference of PanGu Draw 3.0 model for text-to-image generation.


## Getting Started with PanGu Draw 3.0

### Installation

Please make sure the following frameworks are installed.

- python >= 3.7
- mindspore >= 2.2.10  [[install](https://www.mindspore.cn/install)]

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

### Pretrained Weights

The text-to-image task of the Pangu model requires pre-training parameters for both the low timestamp model and the high timestamp model (The pre-training model parameters is coming soon).

| **Version** | **MindSpore Checkpoint** |
|-------------|--------------------------|
| Pangu3-low-timestamp-model | [pangu_low_timestamp-127da122.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/PanGu-Draw-v3/pangu_low_timestamp-127da122.ckpt) |
| Pangu3-high-timestamp-model | [pangu_high_timestamp-c6344411.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/PanGu-Draw-v3/pangu_high_timestamp-c6344411.ckpt) |


### Inference

After obtaining the weights, start the demo using:

```shell
# run txt2img on Ascend
export MS_PYNATIVE_GE=1
python demo/pangu/pangu_sampling.py \
--device_target "Ascend" \
--ms_amp_level "O2" \
--config "configs/inference/pangu_sd_xl_base.yaml" \
--high_solution \
--weight "path/to/low_timestamp_model.ckpt" \
--high_timestamp_weight "path/to/high_timestamp_model.ckpt" \
--prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```


## Examples

Note: sampled 40 steps by PanGu Draw 3.0 on Ascend 910*.

<div align="center">
<img src="https://github.com/Stability-AI/generative-models/assets/143256262/0a8b0b1a-3b54-4a35-beec-4163be61fa88" width="320" />
<img src="https://github.com/Stability-AI/generative-models/assets/143256262/d14a3761-47ad-4671-adcd-866d9d6e6def" width="320" />
<img src="https://github.com/Stability-AI/generative-models/assets/143256262/42e18b39-5c62-4d0f-aa3b-8c0c6e0715f2" width="320" />
</div>
<p align="center">
<font size=3>
<em> Fig1: "一幅中国水墨画：一叶轻舟漂泊在波光粼粼的湖面上，舟上的人正在饮酒放歌" </em> <br>
<em> Fig2: "坐在海边看海浪的少年，黄昏" </em> <br>
<em> Fig3: "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" </em> <br>
</font>
</p>
<br>
