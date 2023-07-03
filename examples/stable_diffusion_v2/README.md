# Stable Diffusion

This folder contains Stable Diffusion models implemented with MindSpore. It targets full support for inference, finetuning, and training from scratch. New checkpoints and features will be continuously updated

## Features
- [x] SD1.4 inference, support Chinese prompt. Though runnable with English prompts, the generation quality is much worse than CN prompts.
- [x] SD1.4 finetune, support Chinese image-text pair data.
- [x] SD2.0 inference, support English prompt. It does not support Chinese prompts.
- [x] SD2.0 finetune, support English image-text par data.
- [x] Support [LoRA finetuning](lora_finetune.md) 🔥 
- [x] Support FID evaluation.

## Quick Start
Please refer to [demo](demo.md) for a quick tour.

## Preparation

### Environment and Dependency

**Device:** Ascend 910

**Framework:** ms1.9, ms2.0rc1 (tested)

Install dependent packages by:
```shell
pip install -r requirements.txt
```

### Pretrained Checkpoint

- SD2.0 
  Download the [SD2.0 checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/stablediffusion/stablediffusionv2_512.ckpt) and put it under `models/` folder 

- SD1.x
Download the [SD1.x checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) (credit to WuKongHuaHua) and put it under `models/` folder

### Dataset for Finetuning (optional)

Prepare image-caption pair data in the follow format

```text
data_path
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

img_txt.csv is the annotation file in the following format
```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

You may download the **pokemon-blip-caption dataset** in [
pokemon_raw.zip](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), which contains 833 pokemon-style images with BLIP-generated captions and is converted to the above format for training. 

- - -
## Stable Diffusion 2.0 - EN
### Inference

```shell
# Text to image generation with SD2.0 
python text_to_image.py --prompt "A wolf in winter"
```

For more argument usages, please run `python text_to_image.py -h`.

For the use of more schedulers/samplers, please refer to the information of [Schedulers](schedulers.md).

### Vanilla Finetuning

Vanilla finetuning refers to the second-stage training in the LDM paper. Only the latent diffusion model (**UNet** + ddpm) will be trained and updated, while CLIP and AutoEncoder are frozen.  

```shell
sh scripts/run_train_v2.sh
```

Modify `data_path` in `run_train_v2.sh` to the path to the dataset that you want to train on. 

### LoRA Finetuning 🔥 

LoRA finetuning has lower memory requirement and allows finetuning on images with higher-resolution such as 768x768.

Please refer to the tutorial of [LoRA for Stable Diffusion Finetuning](lora_finetune.md)


### Evaluation

Please refer to [Evaluation for Diffusion Models](eval/README.md) 

- - -
## Stable Diffusion 1.x - CN


### Inference

```shell
# Text to image generation with SD1.x (Support Chinese) 
python text_to_image.py --prompt "雪中之狼"  -v 1.x
```
> -v is used to set stable diffusion version.

For more argument usages, please run `python text_to_image.py -h`.

### Vanilla Finetuning

```shell
sh scripts/run_train_v1.sh
```

Modify `data_path` in `run_train_v2.sh` to the path to the dataset that you want to train on. 


## What's New
- 2023.06.30  Add LoRA finetuning and FID evalution.
- 2023.06.12  Add velocity parameterization for DDPM prediction type. Usage: set `parameterization: velocity` in configs/your_train.yaml  


## TODO
- [ ] Fix warnings in loading pretrained checkpoints 
- [ ] Support SD2.1 inference and finetuning in 768x768 resolution.
- [ ] Support training from scratch including first-stage training.
