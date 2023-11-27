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
    - [Text-guided Depth-to-Image](#text-guided-depth-to-image)
  - [Training](#training)
    - [LoRA](#efficient-finetuning-with-lora-)
    - [Dreambooth](#dreambooth)
    - [Text Inversion](#text-inversion)
    - [Vanilla Finetuning](#vanilla-finetuning)
    - [v-prediction Finetuning](#v-prediction-finetuning)
    - [Chinese Prompt Adaptation](#chinese-prompt-adaptation)
- [Stable Diffusion 1.5](#stable-diffusion-15)
  - [Inference](#inference-1)
    - [Text-to-Image Generation](#sd15-text-to-image-generation)
    - [Chinese Text-to-Image Generation](#chinese-text-to-image-generation)
    - [Chinese Text-guided Image Inpainting](#chinese-text-guided-image-inpainting)
  - [Training](#training-1)
- [Stable Diffusion with ControlNet](#stable-diffusion-with-controlnet)
  - [Inference](#inference-2)
- [Stable Diffusion with T2I-Adapter](#stable-diffusion-with-t2i-adapter)
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

Currently, we provide pre-trained Stable Diffusion model weights that are compatible with MindSpore as follows.

| **Version name**   | **Task**         | **MindSpore Checkpoint**                                                                                                          | **Ref. Official Model**                                                                           | **Resolution** |
|--------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------|
| 2.1                | text-to-image    | [sd_v2-1_base-7c8d09ce.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_base-7c8d09ce.ckpt)          | [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)         | 512x512        |
| 2.1-v              | text-to-image    | [sd_v2-1_768_v-061732d1.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_768_v-061732d1.ckpt)        | [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)                   | 768x768        |
| 2.1-unclip-l       | image-to-image   | [sd21-unclip-l-baa7c8b5.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd21-unclip-l-baa7c8b5.ckpt)        | [stable-diffusion-2-1-unclip](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip)     | 768x768        |
| 2.1-unclip-h       | image-to-image   | [sd21-unclip-h-6a73eca5.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd21-unclip-h-6a73eca5.ckpt)        | [stable-diffusion-2-1-unclip](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip)     | 768x768        |
| 2.0                | text-to-image    | [sd_v2_base-57526ee4.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt)              | [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)             | 512x512        |
| 2.0-v              | text-to-image    | [sd_v2_768_v-e12e3a9b.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt)            | [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)                       | 768x768        |
| 2.0-inpaint        | image inpainting | [sd_v2_inpaint-f694d5cf.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_inpaint-f694d5cf.ckpt)        | [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | 512x512        |
| 1.5                | text-to-image    | [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt)                    | [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)                    | 512x512        |
| 1.5-wukong         | text-to-image    | [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt)                 |                                                                                                   | 512x512        |
| 1.5-wukong-inpaint | image            | [wukong-huahua-inpaint-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-inpaint-ms.ckpt) |                                                                                                   | 512x512        |

> Resolution refers to the image resolution used in training and is also the optimal choice for image generation.
> Other resolutions are supported (only if divisible by 64) but may lead to a degraded generation quality.
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
python text_to_image.py --prompt "elven forest" -v 2.0
```
> The default version of SD model used is 2.1. It is easy to change the model version by setting the `-v` argument according to the version names defined in [pretrained weights](#pretrained-weights).

For example, to use SD 2.1-v for generating images of 768x768 resolution, please run
```shell
# Text to image generation with SD 2.1-v
python text_to_image.py --prompt "elven forest" -v 2.1-v --H 768 --W 768
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
python inpaint.py \
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
<em> Image inpainting. From left to right: input image, mask, generated images </em>
</p>

### Text-guided Image-to-Image

This pipeline uses a finetuned version of Stable Diffusion 2.1, which can be used to create image variations (image-to-image). The pipeline comes with two pre-trained models, `2.1-unclip-l` and `2.1-unclip-h`, which use the pretrained CLIP Image embedder and OpenCLIP Image embedder separately. You can use the `-v` argument to decide which model to use. The amount of image variation can be controlled by the noise injected to the image embedding, which can be input by the `--noise_level` argument. A value of 0 means no noise, while a value of 1000 means full noise.

You can use the following command to run the image-to-image pipeline

```shell
python unclip_image_variation.py \
    -v {model version} \
    --image_path {path to input image} \
    --prompt "your magic prompt to run image variation."
```
> For more argument usage, please run `python unclip_image_variation.py --help`

Example:

Using `2.1-unclip-l` model as an example,  Please download [sd21-unclip-l-baa7c8b5.ckpt)](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd21-unclip-l-baa7c8b5.ckpt) and [ViT-L-14_stats-b668e2ca.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/unclip/ViT-L-14_stats-b668e2ca.ckpt) to `models/` folder.

And download the [example image](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png) to the running path. Then execute

```shell
python unclip_image_variation.py -v 2.1-unclip-l --image_path tarsila_do_amaral.png --prompt "a cute cat sitting in the garden"
```

The output images are saved in `output/samples` directory.

you can also add extra noise to the image embedding to increase the amount of variation in the generated images.

```shell
python unclip_image_variation.py -v 2.1-unclip-l --image_path tarsila_do_amaral.png --prompt "a cute cat sitting in the garden" --noise_level 200
```

<div align="center">
<img src="https://github.com/zhtmike/mindone/assets/8342575/393832cf-803a-4745-9fb1-7ef1107f9c37" width="960" />
</div>

For Image-to-Image finetuning, please refer to the tutorial of [Stable Diffusion unCLIP Finetuning](unclip_finetune.md) for detailed instructions.

### Text-guided Depth-to-Image

This pipeline allows you to generate new images conditioning on a depth map (preserving image structure) and a text prompt. If you pass an initial image instead of a depth map, the pipeline will automatically extract the depth from it (using Midas depth estimation model) and generate new images conditioning on the image depth, the image, and the text prompt.

It is easy to run with the `depth_to_image.py` script.
```shell
# depth to image conditioning on an input image and text prompt
python depth_to_image.py --prompt {text prompt} \
    --image {path to initial image} \
    --strength 0.7
```
> `--strength` indicates how strong the pipeline will transform the initial image. A lower value - preserve more content of input image. 1 - ignore the initial image and only condition on the depth and text prompt.

```shell
# depth to image given a depth image and text prompt
python depth_to_image.py --prompt {text prompt} --depth_map {path to depth map}
```

Example:

Download the [two-cat image](http://images.cocodataset.org/val2017/000000039769.jpg) and save it in the current folder. Then execute

```shell
python depth_to_image.py --image 000000039769.jpg --prompt "two tigers" --negative_prompt "bad, deformed, ugly, bad anatomy" \
```

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/fa070832-d53f-4bd5-84af-ce8086f41866" width="1024"
 />
</div>
<p align="center">
<em> Text-guided depth-to-image. From left to right: input image, estimated depth map, generated images </em>
</p>

Now, the two cats are replaced with two tigers while the background and image structure are mostly preserved in the generated images.


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

To run vanilla finetuning on a single device, please download [sd_v2_base-57526ee4.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) to `models/` folder, and execute:

```shell
python train_text_to_image.py \
    --train_config "configs/train/train_config_vanilla_v2.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --output_path "output/finetune_pokemon/txt2img" \
    --pretrained_model_path "models/sd_v2_base-57526ee4.ckpt"
```
after setting `data_path` to your dataset path.

> Note: to modify other important hyper-parameters, please refer to training config file `train_config_vanilla_v2.yaml`.


To run in the distributed mode, please execute:
```shell
bash scripts/run_train_v2_distributed.sh
```
after updating `data_path` and `num_devices`, `rank_table_file`, `CANDIDATE_DEVICE` according to your running devices.

**Flash Attention**: You can enable flash attention to reduce the memory footprint. Make sure you have installed MindSpore >= 2.1 and set `enable_flash_attention: True` in `configs/v2-train.yaml`.

To make the text encoder also trainable, please set `cond_stage_trainable: True` in `configs/v2-train.yaml`

### v-prediction Finetuning

The default objective used in SD training is to minimize the noise prediction error (noise-prediction). To alter the objective to v-prediction, which is used in SD 2.0-v training, please refer to [v-prediction.md](v_prediction.md)

### Chinese Prompt Adaptation

To make SD work better with Chinese prompts, one can replace the default text encoder with [CN-CLIP](https://github.com/ofa-sys/chineseclip) and run [vanilla finetuning](#vanilla-finetuing) on a Chinese text-image pair dataset.

CN-CLIP is an open-source CLIP implementation that is trained on an extensive dataset of Chinese text-image pairs. The main difference between CN-CLIP and OpenCLIP is the tokenizer and the first embedding layer.

To replace the original CLIP used in SD with CN-CLIP, please:

1. Download [CN-CLIP ViT-H/14](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/ms_cnclip_h14-d780480a.ckpt) to `models` folder.

2. Run the vanilla training script after setting the `--custom_text_encoder` and `--model_config` arguments.

```shell
python train_text_to_image.py --custom_text_encoder models/ms_cnclip_h14-d780480a.ckpt --model_config configs/v2-inference-cnclip.yaml ...
```

After the training is finished, similarly, you can load the model and run Chinese text-to-image generation.

```shell
python text_to_image.py --config configs/v2-inference-cnclip.yaml --ckpt_path {path to trained checkpoint} ...
```

- - -
# Stable Diffusion 1.5

## Inference

It is simple to switch from SD 2.0 to SD 1.5 by setting the `--version` (`-v`) argument.

### SD1.5 Text-to-Image Generation

Download [SD1.5 checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) to `models/` folder. Then run,

```shell
python text_to_image.py --prompt "A cute wolf in winter forest" -v 1.5
```

### Chinese Text-to-Image Generation

Download [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) to `models/` folder. Then run,

```shell
python text_to_image.py --prompt "雪中之狼"  -v 1.5-wukong
```

### Chinese Text-guided Image Inpainting

Download [wukong-huahua-inpaint-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-inpaint-ms.ckpt) to `models/` folder. Then run,

```shell
python inpaint.py --image {path to input image} --mask {path to mask image} --prompt "图片编辑内容描述"  -v 1.5-wukong
```

## Training

To train SD 1.5 on a custom text-image dataset, please download [SD1.5 checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) to `models/` folder, and execute:

```shell
# SD 1.5 vanilla training
python train_text_to_image.py \
    --train_config "configs/train/train_config_vanilla_v1.yaml" \
    --data_path "datasets/pokemon_blip/train" \
    --output_path "output/finetune_pokemon/txt2img" \
    --pretrained_model_path "models/sd_v1.5-d0ab7146.ckpt"
```
after setting `data_path` to your dataset path.

> Note: to modify other important hyper-parameters, please refer to training config file `train_config_vanilla_v1.yaml`.


To train SD 1.5 on a Chinese text-image pair dataset, please download [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) to `models/` folder, and execute:

```shell
python train_text_to_image.py \
    --train_config "configs/train/train_config_vanilla_v1_chinese.yaml" \
    --data_path "datasets/pokemon_cn/train" \
    --output_path "output/txt2img" \
    --pretrained_model_path "models/wukong-huahua-ms.ckpt"
```

# Stable Diffusion with ControlNet

ControlNet controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

## Inference

For details, please refer to the tutorial [ControlNet image generation](controlnet.md).

# Stable Diffusion with T2I-Adapter

[T2I-Adapter](../t2i_adapter/README.md) is a simple and lightweight network that provides extra visual guidance for
Stable Diffusion models without re-training them. The adapter act as plug-ins to SD models, making it easy to integrate
and use.

For more information on inference and training with T2I-Adapters, please refer
to [T2I-Adapter](../t2i_adapter/README.md) page.

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

Please refer to [Evaluation for Diffusion Models](tools/eval/README.md)


- - -

## What's New

- 2023.08.30
  - Add T2I-Adapter support for text-guided Image-to-Image translation.
- 2023.08.24
  - Add Stable Diffusion v2.1 and v2.1-v (768)
  - Support checkpoint auto-download
- 2023.08.17
  - Add Stable Diffusion v1.5
  - Add DreamBooth fine-tuning
  - Add text-guided image inpainting
  - Add CLIP score metrics (CLIP-I, CLIP-T) for evaluating visual and textual fidelity
- 2023.07.05
  - Add negative prompts
  - Improve logger
  - Fix bugs for MS 2.0.
- 2023.06.30
  - Add LoRA fine-tuning and FID evaluation.
- 2023.06.12
  - Add velocity parameterization for DDPM prediction type. Usage: set `parameterization: velocity` in
    configs/your_train.yaml

## Contributing
We appreciate all kinds of contributions, including making **issues** or **pull requests** to make our work better.
