# SDXL的离线推理

## 安装指南

⚠️注意：MindSpore Lite采用Python 3.7环境，请在安装前确保已配置好Python 3.7环境。

⚠️注意：MindSpore与MindSpore Lite版本需保持一致。

### 1. 安装MindSpore

请按照[MindSpore官方安装教程](https://www.mindspore.cn/install)安装MindSpore 2.1版本。

### 2. 安装MindSpore Lite

参照[MindSpore Lite官方文档](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/downloads.html)进行安装：

1. 根据运行环境下载对应的tar.gz和whl压缩包。
   
   ```shell
   # 解压并安装MindSpore Lite相应版本的WHL包
   tar -zxvf mindspore-lite-2.1.0-*.tar.gz
   pip install mindspore_lite-2.1.0-*.whl
   
## 准备工作
### 1. 转换预训练权重（`.safetensors` -> `.ckpt`）

我们提供了一个脚本，用于将预训练权重从 `.safetensors` 转换为 `.ckpt`，位于 `tools/model_conversion/convert_weight.py`。

**步骤1：** 从 [Official](https://github.com/Stability-AI/generative-models) 下载预训练权重 [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)。

**步骤2：** 将权重转换为 MindSpore 的 `.ckpt` 格式，并放置在 `./models/` 目录下。

```shell
# 转换 sdxl-base-1.0 模型
cd tools/model_conversion
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml
```

