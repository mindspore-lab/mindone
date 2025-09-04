# SAM 2：图像与视频的任意分割

[English](README.md) | **中文**

这是 Meta, FAIR 的 [SAM2](https://github.com/facebookresearch/sam2) 在 MindSpore 框架下的实现代码。

Segment Anything Model 2（SAM 2）是一个用于图像/视频视觉分割任务的基础模型。其模型架构为带有流式记忆的 Transformer，支持实时视频处理。详细架构如下图所示。

![SAM2](https://github.com/facebookresearch/sam2/blob/main/assets/model_diagram.png?raw=true)

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

- 📍 目前**SAM2** 具备以下功能：
    - ✅ 给定静态图像和参考点，预测分割掩码。
    - ✅ 给定静态图像，预测分割掩码。

### TODO
* [ ] 视频输入的推理脚本 **[开发中]**
* [ ] 训练脚本 **[开发中]**
* [ ] 基准测试 **[开发中]**

欢迎您的贡献！

## 🚀 快速开始

### 权重文件

请使用以下命令下载权重文件：
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### 单掩码预测

`predict_image.py` 脚本可根据输入图像和参考点推理分割掩码。请先下载用于推理的图片：
```
mkdir images
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg
```
然后运行 `python predict_image.py --image_path images/truck.jpg` 和 `python predict_image.py --image_path images/groceries.jpg`。结果如下，会保存在`images/`文件夹下。

| 原图（参考点用Star标注） |  预测结果1 |  预测结果2 |  预测结果3 |
|:--------------------------:|:-------------------------------------:|:-------------------------------------:|:-------------------------------------:|
| ![原图1](images/truck.jpg) | ![预测结果1](images/truck_result1.jpg) | ![预测结果2](images/truck_result2.jpg) | ![预测结果3](images/truck_result3.jpg) |
| ![原图2](images/groceries.jpg) | ![预测结果1](images/groceries_result1.jpg) | ![预测结果2](images/groceries_result2.jpg) | ![预测结果3](images/groceries_result3.jpg) |


### 分割掩码预测

你可以使用 `predict_mask.py` 从给定图像获取分割掩码。请先下载用于推理的图片：
```bash
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/cars.jpg
```
然后运行 `python predict_mask.py`。
