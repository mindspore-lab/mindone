# Stable Diffusion

This folder contains Stable Diffusion (SD) models implemented with MindSpore. It targets at full support for inference, finetuning, and training from scratch. New checkpoints and features will be continuously updated

## Features
- [x] Text-to-image generation based on Stable Diffusion 2.0.
- [x] Support SoTA diffusion process schedulers including DDIM, DPM Solver, UniPC, etc. (under continuous update)
- [x] Vanilla Stable Diffusion finetuning
- [x] [Efficient SD finetuning with LoRA](lora_finetune.md) 🔥
- [x] [Finetuning with DreamBooth](dreambooth_finetune.md)
- [x] Quantitative evaluation for diffusion models: FID
- [x] Chinese text-to-image generation thanks to Wukonghuahua (based on SD 1.x)
- [x] Negative prompt guidance.

For a quick tour, please view [demo](demo.md).

## Installation & Preparation

### Environment and Dependency

**Device:** Ascend 910

**Framework:** MindSpore >= 1.9

Install dependent packages by:
```shell
pip install -r requirements.txt
```

### Pretrained Checkpoint

- SD2.0
  Download [SD2.0 checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) and put it under `models/` folder

- SD1.x (Chinese)
Download [SD1.x checkpoint](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) (credit to WuKongHuaHua) and put it under `models/` folder


### Dataset Preparation for Finetuning (Optional)

The text-image pair dataset for finetuning should follow the file structure below

```text
dir
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

For convenience, we have prepared two public text-image datasets obeying the above format.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.
- [Chinese-art blip caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 100 chinese art-style images with BLIP-generated captions.

To use them, please download `pokemon_blip.zip` and `chinese_art_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets). Then unzip them on your local directory, e.g. `./datasets/pokemon_blip`.


- - -
## Stable Diffusion 2.0

### 1. Text-to-Image Generation

```shell
# Text to image generation with SD2.0
python text_to_image.py --prompt "elven forest"
```
For more argument usages, please run `python text_to_image.py -h`.

#### 1.1 Negative Prompt Guidance

While `--prompt` indicates what to render in the generated images, the negative prompt (`--negative_prompt`) can be used to tell Stable Diffusion what you don't want to see in the generated images. It can be useful in reducing specific artifacts. Here is an example for removing 'moss' from the 'elven forest':

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/1c35853d-036f-459c-944c-9953d2da8087" width="320" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/SamitHuang/mindone/assets/8156835/b1f037ca-4e03-40e4-8da2-d358801eadd5)" width="320" />
</div>
<p align="center">
  <em> Prompt: "elven forest"</em>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <em> With negative prompt: "moss" </em>
</p>

#### 1.2 Supported Diffusion Process Schedulers

- DDIM
- DPM Solver
- DPM Solver++
- PLMS
- UniPC

For detailed usage of the schedulers/samplers, please refer to [Diffusion Process Schedulers](schedulers.md).

### 2. Vanilla Finetuning

Vanilla finetuning refers to the second-stage training in the LDM paper. Only the latent diffusion model (**UNet** + ddpm) will be trained and updated, while CLIP and AutoEncoder are frozen.

```shell
sh scripts/run_train_v2.sh
```

Modify `data_path` in `run_train_v2.sh` to the path to the dataset that you want to train on.

For training on large datasets, please use the distributed training script via
```
bash scripts/run_train_v2_distributed.sh
```
, after modifying the data paths and device nums in the script.

### 3. Efficient Finetuning with LoRA 🔥

LoRA finetuning has lower memory requirement and allows finetuning on images with higher-resolution such as 768x768.

Please refer to the tutorial of [LoRA for Stable Diffusion Finetuning](lora_finetune.md)

### 4. Finetuning with DreamBooth

DreamBooth allows users to generate contextualized images of one subject using just 3-5 images of the subject, e.g., your dog.

Please refer to the tutorial of [DreamBooth for Stable Diffusion Finetuning](dreambooth_finetune.md)

### 5. Evaluation

Please refer to [Evaluation for Diffusion Models](eval/README.md)

- - -
## Stable Diffusion 1.x

### 1. Chinese Text-to-Image Generation

```shell
# Text to image generation with SD1.x (Support Chinese)
python text_to_image.py --prompt "雪中之狼"  -v 1.x
```
> -v is used to set stable diffusion version.

For more argument usages, please run `python text_to_image.py -h`.

### 2. Vanilla Finetuning

```shell
sh scripts/run_train_v1.sh
```

Modify `data_path` in `run_train_v2.sh` to the path to the dataset that you want to train on.


## What's New
- 2023.07.05  Add negative prompts; Improve logger; Fix bugs for MS 2.0.
- 2023.06.30  Add LoRA finetuning and FID evalution.
- 2023.06.12  Add velocity parameterization for DDPM prediction type. Usage: set `parameterization: velocity` in configs/your_train.yaml


## Contributing
We appreciate all kinds of contributions including making **issues** or **pull requests** to make our work better.
