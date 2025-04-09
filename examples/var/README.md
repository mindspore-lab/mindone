# VAR based on MindSpore

This repository is the MindSpore implementation of [VAR](https://arxiv.org/abs/2404.02905)[<a href="#references">1</a>].

VAR, a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine “next-scale prediction” or “next-resolution prediction”, diverging from the standard raster-scan “next-token prediction”.

<p align="center">
  <img width="430" alt="Image" src="https://github.com/user-attachments/assets/1e1024b4-61b4-49a8-9628-bda3ea4bd6ea" />
</p>
<p align="center">
  <em> Figure 1. Standard autoregressive modeling (AR) vs. visual autoregressive modeling (VAR) [<a href="#references">1</a>] </em>
</p>


## Requirements

| mindspore | ascend driver | firmware    | cann toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
| 2.5.0     | 24.1.RC3    | 7.3.0.1.231 | 8.0.RC3.beta1       |

```shell
cd mindone
pip install -e .[training]
cd examples/var
pip install -r requirements.txt
```

## Traning

### data prepare
Dataset should be a tree of directories. It should be like this:
```
/path/to/dataset/
├── train
    ├── class_1
        ├──image_1.jpg
        ├──image_2.jpg
        ├──...
    ├── class_2
        ├──image_1.jpg
        ├──image_2.jpg
        ├──...
├── val
    ├── class_1
        ├──image_1.jpg
        ├──image_2.jpg
        ├──...
    ├── class_2
        ├──image_1.jpg
        ├──image_2.jpg
        ├──...
```
### model wight
We provide weight conversion script `tools/convert_weight.py` to convert the original Pytorch model weights to MindSpore model weights. Pytorch model weights can be accessed via links below.

| Model              | Model size | URL                                                             |
|--------------------|------------|-----------------------------------------------------------------|
| VAE                | 109M       |  [Download](https://huggingface.co/FoundationVision/var/blob/main/vae_ch160v4096z32.pth) |
| VAR-d16           | 310M       |  [Download](https://huggingface.co/FoundationVision/var/blob/main/var_d16.pth) |

```shell
python tools/convert_weight.py --src /path/to/var-d16.pth --target model/var-d16.ckpt
python tools/convert_weight.py --src /path/to/vae_ch160v4096z32.pth --target model/vae-2972c247.ckpt
```

### Finetuning
```shell
bash script/run_train_d16.sh
```

### Performance

Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 pynative mode.

| model name    |  cards          | batch size      | resolution   |  FA   |  precision |  jit level | graph compile |s/step     | img/s |
|:-------------:|:------------:|:------------:|:-----------------------:|:------------:|:------------------:|:----------------:|:----------------:|:----------------:|:----------------:|
| VAR-d16 |  1               | 96               | 256x256  | ON | fp16 | / | / |  1.70 | 56.47 |
| VAR-d16 |  2               | 32               | 256x256  | ON | fp16 | / | / |  0.71 | 90.14 |
| VAR-d16 |  2               | 96               | 256x256  | ON | fp16 | / | / |  1.75 | 109.71 |


## Inference

Here are a inference demo.
```shell
python inference_demo.py --var_ckpt path/to/var-d16.ckpt
```

### Performance

Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 pynative mode.

| model name    |  cards          | batch size      | resolution   |  FA   |  precision |  jit level | graph compile | s/img |
|:-------------:|:------------:|:------------:|:-----------------------:|:------------:|:------------------:|:------------------------:|:----------------:|:----------------:|
| VAR-d16 |  1              | 8               | 256x256  | ON | fp16 | / | / | 0.32 |

Here are some inference results


<p float="center">
<img alt="Image" src="https://github.com/user-attachments/assets/5a8da3e4-93bb-4f1d-b8c3-ddda290c5bcd" />
</p>


## References

[1] Tian K, Jiang Y, Yuan Z, et al. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction[J]. Advances in neural information processing systems, 2023, 37: 84839-84865.
