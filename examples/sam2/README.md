# SAM 2: Segment Anything in Images and Videos

This is a MindSpore Implementation of [SAM2](https://github.com/facebookresearch/sam2) from Meta, FAIR.


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
    - ✅ Prediction of sgementation masks given a static image.


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

### Image Prediction

`predict_image.py` is a script to infer the mask given an input image and a reference point. Please download the images for inference first:
```
mkdir images
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg
```
Then run `python predict_image.py`.

### Mask Prediction

You can use `predict_mask.py` to get the segmentation mask from a given image. Please download the images for inference first:
```bash
wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/cars.jpg
```
Then run `python predict_mask.py`.
