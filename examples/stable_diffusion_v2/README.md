# Stable Diffusion

This folder contains various [Stable Diffusion](https://arxiv.org/abs/2112.10752) (SD) models and pipelines implemented with MindSpore. It targets at full support for inference, finetuning, and training from scratch. New models and features will be continuously updated.

## Features

- [x] Text-to-image generation based on Stable Diffusion 2.0.
- [x] Support SoTA diffusion process schedulers including DDIM, DPM Solver, UniPC, etc. (under continuous update)
- [x] Vanilla Stable Diffusion finetuning
- [x] [Efficient SD finetuning with LoRA](lora_finetune.md) 🔥
- [x] [Finetuning with DreamBooth](dreambooth_finetune.md)
- [x] Quantitative evaluation for diffusion models: FID, CLIP Scores (CLIP-I, CLIP-T)
- [x] Chinese text-to-image generation thanks to Wukonghuahua
- [x] Negative prompt guidance.

For a quick tour, please view [demo](demo.md).

## Usage

- [Installation](#installation)
- [Pretrained Weights](#pretrained-weights)
- [Stable Diffusion 2.0](#stable-diffusion-20)
  - [Inference](#inference)
    - [Text-to-Image Generation](#text-to-image-generation)
    - [Text-guided Image Inpainting](#text-guided-image-inpainting)
    - [Text-guided Image-to-Image](#text-guided-image-to-image)
  - [Training](#training)
    - [LoRA](#efficient-finetuning-with-lora-)
    - [Dreambooth](#dreambooth)
    - [Text Inversion](#text-inversion)
    - [Vanilla Finetuning](#vanilla-finetuning)
    - [v-prediction Finetuning](#v-prediction-finetuning)
    - [Chinese Prompt Adaptation](#chinese-prompt-adaptation)
- [Stable Diffusion 1.5](#stable-diffusion-15)
  - [Inference](#inference-1)
  - [Training](#training-1)
- [Data Preparation for Training](#dataset-preparation-for-finetuning)
- [Supported Schedulers](#supported-schedulers)
- [Evaluation](#evaluation)


## Installation

Please make sure the following frameworks are installed.

- mindspore >= 1.9  [[install](https://www.mindspore.cn/install)] (2.0 is recommended for the best performance.)
- python >= 3.7

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

## Pretrained Weights
<!---
<details close>
  <summary>Pre-trained SD weights that are compatible with MindSpore: </summary>
-->

Currently, we provide pre-trained stable diffusion model weights that are compatible with MindSpore as follows.

| **Version name** |**Task** |  **MindSpore Checkpoint**  | **Ref. Official Model** | **Resolution**|
|-----------------|---------------|---------------|------------|--------|
| 2.1      | text-to-image | [sd_v2-1_base-7c8d09ce.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_base-7c8d09ce.ckpt) |  [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) | 512x512 |
| 2.1-v      | text-to-image | [sd_v2-1_768_v-061732d1.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_768_v-061732d1.ckpt) |  [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | 768x768 |
| 2.0            | text-to-image | [sd_v2_base-57526ee4.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) |  [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) | 512x512 |
| 2.0-v      | text-to-image | [sd_v2_768_v-e12e3a9b.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt) |  [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) | 768x768 |
| 2.0-inpaint      | image inpainting | [sd_v2_inpaint-f694d5cf.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_inpaint-f694d5cf.ckpt) | [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | 512x512|
| 1.5       | text-to-image | [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) | [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | 512x512 |
| wukong    | text-to-image |  [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) |  | 512x512 |
| wukong-inpaint    | image |  [wukong-huahua-inpaint-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-inpaint-ms.ckpt) |  | 512x512 |

> Resolution refers to the image resolution used in training. Although it is also the default choice for generation. you can still change the target height or width to generate images of various shape (H and W must be divisable by 64) 
<!---
</details>
-->

To transfer other Stable Diffusion models to MindSpore, please refer to [model conversion](tools/model_conversion/README.md).

- - -
# Stable Diffusion 2.0

## Inference

### Text-to-Image Generation

To generate images by providing a text prompt, please download [sd_v2_base-57526ee4.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) to `models/` folder, and run:

```shell
# Text to image generation with SD-2.0-base
python text_to_image.py --prompt "elven forest"
```
> The default version of SD model used is 2.0. It is easy to switch to another SD version by setting the `-v` argument according to the version names defined in [pretrained weights](#pretrained-weights) and downloading the target checkpoint.

For example, you may switch to sd 2.0 v768 to generate images in 768x768 resolution by
```shell
# Text to image generation with SD-2.0-v768
python text_to_image.py --prompt "elven forest" -v 2.0_v768
```

For more argument usages, please run `python text_to_image.py -h`.

#### Negative Prompt Guidance

While `--prompt` indicates what to render in the generated images, the negative prompt (`--negative_prompt`) can be used to tell Stable Diffusion what you don't want to see in the generated images. It can be useful in reducing specific artifacts. Here is an example of removing 'moss' from the 'elven forest':

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

### Text-guided Image Inpainting

Text-guided image inpainting allows users to edit specific regions of an image by providing a mask and a text prompt, which is an interesting erase-and-replace editing operation. When the prompt is set to empty, it can be applied to auto-fill the masked regions to fit the image context (which is similar to the AI fill and extend operations in PhotoShop-beta).

Please download [sd_v2_inpaint-f694d5cf.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_inpaint-f694d5cf.ckpt)  to `models/` folder, and execute:

```shell
python inpaint.py
    --image {path to input image} \
    --mask  {path to mask image} \
    --prompt "your magic prompt to paint the masked region"
```
> For more argument usage, please run `python inpaint.py --help`

Example:

Download the [example image](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png) and [mask](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png) to the running path. Then execute

```shell
python inpaint.py  --image overture-creations-5sI6fQgYIuo.png --mask overture-creations-5sI6fQgYIuo_mask.png \
    --prompt "Face of a yellow cat, high resolution, sitting on a park bench"
```

Now the masked region is smoothly replaced with the instructed content.
<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/f0d6073e-fe24-4d3d-8f54-b7c4833bb206" width="960" />
</div>
<p align="center">
<em> Text-guided image inpainting. From left to right: input image, mask, generated images. </em>
</p>

By setting empty prompt (`--prompt=""`), the masked part will be auto-filled to fit the context and background.
<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/21158de6-b9ec-4538-83cf-2a3bbea649e7" width="960"
 />
</div>
<p align="center">
<em> Image inpainting. (From left to right: input image, mask, generated images) </em>
</p>

### Text-guided Image-to-Image

Coming soon


## Training

To create a dataset for training, please refer to [data preparation](#dataset-preparation-for-finetuning).

### Efficient Finetuning with LoRA 🔥

Low-Rank Adaptation (LoRA) is a parameter-efficient finetuning method for large models.

Please refer to the tutorial of [LoRA for Stable Diffusion Finetuning](lora_finetune.md) for detailed instructions.


### DreamBooth

DreamBooth allows users to generate contextualized images of one subject using just 3-5 images of the subject, e.g., your dog.

Please refer to the tutorial of [DreamBooth for Stable Diffusion Finetuning](dreambooth_finetune.md) for detailed instructions.


### Text Inversion

Coming soon

### Vanilla Finetuning

Vanilla finetuning is to finetune the latent diffusion model (UNet) directly, which can be viewed as the second-stage training mentioned in the [LDM paper](https://arxiv.org/abs/2112.10752). In this setting, both CLIP-TextEncoder and VAE will be frozen, and **only UNet will be updated**.

To run vanilla finetuning on a single device, please execute:

```shell
sh scripts/run_train_v2.sh
```
after setting `data_path` to your dataset path.


To run in the distributed mode, please execute:
```
bash scripts/run_train_v2_distributed.sh
```
, after updating `data_path` and `num_devices`, `rank_table_file`, `CANDIDATE_DEVICES` according to your running devices.

**Flash Attention**: You can enable flash attention to reduce the memory footprint. Make sure you have installed MindSpore >= 2.1 and set `enable_flash_attention: True` in `configs/v2-train.yaml`.

To make the text encoder also trainable, please set `cond_stage_trainable: True` in `configs/v2-train.yaml`

### v-prediction Finetuning

The default objective used in SD training is to minimize the noise prediction error (noise-prediction). To alter the objective to v-prediction, which is used in SD 2.0-v training, please refer to [v-prediction.md](v_prediction.md)

### Chinese Prompt Adaptation

To make SD work better with Chinese prompts, one can replace the default text encoder with [CN-CLIP](https://github.com/ofa-sys/chineseclip) and run [vanilla finetuning](#vanilla-finetuing) on a Chinese text-image pair dataset.

CN-CLIP is an open-source CLIP implementation that is trained on an extensive dataset of Chinese text-image pairs. The main difference between CN-CLIP and OpenCLIP is the tokenizer and the first embedding layer.

To replace the original CLIP used in SD with CN-CLIP, please:

1. Download [CN-CLIP ViT-H/14](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/ms_cnclip_h14-d780480a.ckpt) to `models` folder.

2. Update `scripts/train_config_v2.json` by setting the model config as follows.

`"model_config": "configs/v2-train-cnclip.yaml"`

3. Run the vanilla training script after setting the `--custom_text_encoder` and `--config` arguments.

```
python train_text_to_image.py --custom_text_encoder models/ms_cnclip_h14-d780480a.ckpt --config configs/v2-inference-cnclip.yaml ...
```

After the training is finished, similarly, you can load the model and run Chinese text-to-image generation.

```
python text_to_image.py --config configs/v2-inference-cnclip.yaml --ckpt_path {path to trained checkpoint} ...
```

- - -
# Stable Diffusion 1.5

## Inference

It is simple to switch from SD 2.0 to SD 1.5 by setting the `--version` (`-v`) argument.

### SD1.5 Text-to-Image Generation

Download [SD1.5 checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) to `models/` folder. Then run,

```
python text_to_image.py --prompt "A cute wolf in winter forest" -v 1.5
```

### Chinese Text-to-Image Generation

Download [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) to `models/` folder. Then run,

```
python text_to_image.py --prompt "雪中之狼"  -v wukong
```

### Chinese Text-guided Image Inpainting

Download [wukong-huahua-inpaint-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-inpaint-ms.ckpt) to `models/` folder. Then run,

```
python inpaint.py --image {path to input image} --mask {path to mask image} --prompt "图片编辑内容描述"  -v wukong
```

## Training

To train SD 1.5 on a custom text-image dataset, please run

```shell
# SD 1.5 vanilla training
sh scripts/run_train_v1.sh
```
after setting `data_path` in `run_train_v1.sh` to your dataset path.

> Note: to run other training pipelines on SD 1.5, you can refer to training tutorials of SD 2.0 and change the following arguments in the training script: set `--model_config` argument to `configs/v1-train.yaml`, `--train_config` to `configs/train_config.json`, and set `--ckpt_path` to `models/sd_v1.5-d0ab7146.ckpt`.


# Dataset Preparation for Finetuning

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


# Supported Schedulers

- DDIM
- DPM Solver
- DPM Solver++
- PLMS
- UniPC

For detailed usage of the schedulers/samplers, please refer to [Diffusion Process Schedulers](schedulers.md).


# Evaluation

Please refer to [Evaluation for Diffusion Models](eval/README.md)


- - -
## What's New
- 2023.08.17
  - Add Stable Diffusion v1.5
  - Add Dreambooth finetuning
  - Add text-guided image inpainting
  - Add CLIP score metrics (CLIP-I, CLIP-T) for evaluating visual and textual fidelity
- 2023.07.05
  - Add negative prompts
  - Improve logger
  - Fix bugs for MS 2.0.
- 2023.06.30
  - Add LoRA finetuning and FID evalution.
- 2023.06.12
  - Add velocity parameterization for DDPM prediction type. Usage: set `parameterization: velocity` in configs/your_train.yaml


## Contributing
We appreciate all kinds of contributions including making **issues** or **pull requests** to make our work better.
