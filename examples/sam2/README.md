# SAM 2: Segment Anything in Images and Videos

**English** | [中文](README_CN.md)

This is a MindSpore Implementation of [SAM2](https://github.com/facebookresearch/sam2) from Meta, FAIR.

Segment Anything Model 2 (SAM 2) is a foundation model for image/video visual segmentation task. The model architecture is a transformer with streaming memory for real-time video processing. Detailed architecture is shown as follows.

![SAM2](https://github.com/facebookresearch/sam2/blob/main/assets/model_diagram.png?raw=true)


## 📦 Requirements


<div align="center">

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.6.0   |  24.1.RC3     | 7.6.0.1.220 |  8.0.RC3.beta1     |

</div>

1. Install
   [CANN 8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1)
   and MindSpore according to the [official instructions](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```
3. Install mindone
    ```
    cd mindone
    pip install -e .
    ```
    Try `python -c "import mindone"`. If no error occurs, the installation is successful.

## 🔆 Features

- 📍 **SAM2** with the following features
    - ✅ Prediction of masks given a staic image and a reference point.
    - ✅ Prediction of segmentation masks given a static image.


### TODO
* [ ] inference script for video input **[WIP]**.
* [ ] training script **[WIP]**.
* [ ] benchmark **[WIP]**.

You contributions are welcome.

## 🚀 Quick Start

### Checkpoints

Please download checkpoints using:
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### Single Mask Prediction

`predict_image.py` is a script to infer the mask given an input image and a reference point. Please download the images for inference first:
```
mkdir images
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg
```
Then run `python predict_image.py`.

The results will be saved in the `images/` folder as shown below:

| Original (reference point marked with Star) | Prediction 1 | Prediction 2 | Prediction 3 |
|:-------------------------------------------:|:-------------:|:-------------:|:-------------:|
| ![truck_point](https://github.com/user-attachments/assets/b9e2b831-35e5-4824-8407-68b51a27b891)| ![truck_infer_0](https://github.com/user-attachments/assets/42baa9a0-4485-4a50-9724-8c30f9a6212d) | ![truck_infer_1](https://github.com/user-attachments/assets/d133cb9a-25b3-4b75-b51c-860ee70a0251) | ![truck_infer_2](https://github.com/user-attachments/assets/4810f0e9-0e56-4fb2-a54c-2e58417d9281)|
| ![groceries_point](https://github.com/user-attachments/assets/df421848-afe5-4dd5-8d35-9f51442515a5)| ![groceries_infer_0](https://github.com/user-attachments/assets/f33fdaa4-2684-4a66-b701-42b795bbd293) | ![groceries_infer_1](https://github.com/user-attachments/assets/b49c11bc-c40d-4fdf-9e73-97b2a358568c) | ![groceries_infer_2](https://github.com/user-attachments/assets/3c72ea9f-5668-4688-a8db-1489984ce2a6) |

### Segmentation Mask Prediction

You can use `predict_mask.py` to get the segmentation mask from a given image. Please download the images for inference first:
```bash
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/cars.jpg
```
Then run `python predict_mask.py`.

The results will be saved in the `images/` folder as shown below:

| Original | Prediction |
|:--------:|:----------:|
| <img src="https://github.com/user-attachments/assets/b422f52b-b10a-401c-be9a-efcad0eb696a" width="300" alt="cars"> | <img src="https://github.com/user-attachments/assets/ca348fe3-481c-41bf-a62f-2c430f2f6972" width="300" alt="cars">  |
