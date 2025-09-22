# Language Segment-Anything

[English](README.md) | **中文**

这是 [Language SAM](https://github.com/luca-medeiros/lang-segment-anything) 的 MindSpore 实现代码。

Language SAM 基于 Meta 的 Segment Anything Model 2 和 GroundingDINO 检测模型。它可以根据文本提示进行目标检测和图像分割。

![Lang_SAM](https://github.com/luca-medeiros/lang-segment-anything/raw/main/assets/outputs/person.png)

## 📦 环境要求

<div align="center">

| MindSpore | Ascend 驱动 | 固件版本 | CANN 工具包/内核 |
|:---------:|:-----------:|:--------:|:----------------:|
|   2.6.0   | 24.1.RC3    | 7.6.0.1.220 | 8.0.RC3.beta1 |

</div>

1. 安装 [CANN 8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) 和 MindSpore，具体请参考[官方安装说明](https://www.mindspore.cn/install)。
2. 安装依赖
    ```shell
    pip install -r requirements.txt
    ```
3. 安装 mindone
    ```
    cd mindone
    pip install -e .
    ```
    测试 `python -c "import mindone"`，无报错即安装成功。

## 🔆 功能特性

- 📍 **Language SAM** 具备以下功能：
    - ✅ 给定静态图像和文本提示，预测分割掩码。

### TODO
* [ ] 批量推理脚本 **[开发中]**
* [ ] Gradio 演示 **[开发中]**

欢迎您的贡献！

## 🚀 快速开始

### 权重文件

请使用以下命令下载权重文件：
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### 图像推理

`predict_image.py` 脚本可根据输入图像和文本提示推理分割掩码。请先下载用于推理的图片：
```bash
mkdir -p assets
wget -P assets https://raw.githubusercontent.com/luca-medeiros/lang-segment-anything/refs/heads/main/assets/car.jpeg
```
然后运行 `python predict_image.py`。
