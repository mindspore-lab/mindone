# Language Segment-Anything

This is a MindSpore Implementation of [Language SAM](https://github.com/luca-medeiros/lang-segment-anything).


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

## 🚀 Quick Start

### Checkpoints

Please download checkpoints using:
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### Image Prediction

`predict_image.py` is a script to infer the mask given an input image and a text prompt. Please download the images for inference first:
```bash
mkdir -p assets
wget -P assets https://raw.githubusercontent.com/luca-medeiros/lang-segment-anything/refs/heads/main/assets/car.jpeg
```
Then run `python predict_image.py`.
