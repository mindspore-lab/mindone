# Stable Diffusion based on MindSpore

<p float="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/46a3b7f1-240d-4933-a2c7-f49ce995ffd5" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/91c5cff0-5a35-4624-98f4-61777fd883a0" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/37f3e414-a1a4-4743-8ad1-5de7c10c3036" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/9e456f4e-8174-467c-8d0c-7a42c9f19dc7" width="25%" />
</p>
<p float="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/86d63435-fb41-423f-8d11-d55185924fd7" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/b2dd903b-aafb-4552-a7fb-3da5f7ae8baa" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/e146e0ff-9b49-4cf6-8458-c2abea29cc6b" width="16.66%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/8994ca17-5d7c-41fe-bfe3-baece43baddc" width="16.66%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/fc9d2149-e3a3-4047-b343-fbdab354d3d0" width="16.66%" />
</p>

## Table of Contents
- [Introduction](#introduction)
    - [Supported Models and Pipelines](#supported-models-and-pipelines) 🔥
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Text-to-Image](#text-to-image)
    - [Inference](#inference)
    - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [LoRA Fine-tuning](#lora-fine-tuning) 🔥
    - [Dreambooth Fine-tuning](#dreambooth-fine-tuning)
    - [Textual Inversion Fine-tuning](#textual-inversion-fine-tuning)
- [Image-to-Image](#image-to-image)
    - [Image Variation](#image-variation)
    - [Inpainting](#inpainting)
    - [Depth-to-Image](#depth-to-image)
- [ControlNet](#controlnet)
- [T2I Adapter](#t2i-adapter)
- [Advanced Usage](#advanced-usage)
    - [Model Conversion](#model-conversion)
    - [Schedulers](#schedulers)
    - [Training with v-prediction](#training-with-v-prediction)
    - [Diffusion Model Evaluation](#diffusion-model-evaluation)
    - [Safety Checker](#safety-checker)
    - [Watermark](#watermark)

## Introduction

This repository integrates state-of-the-art [Stable Diffusion](https://arxiv.org/abs/2112.10752) models including SD1.5, SD2.0, and SD2.1,
supporting various generation tasks and pipelines. Efficient training and fast inference are implemented based on MindSpore.

New models and features will be continuously updated.

<!--
This repository provides "small" but popularly used diffusion models like SD1.5. Currently, we support the following tasks and models.
-->

### Supported Models and Pipelines


| **SD Model**  | **Text-to-Image**      | **Image Variation** | **Inpainting**  | **Depth-to-Image**  | **ControlNet**  |**T2I Adapter**|
|:---------------:|:--------------:|:--------------------:|:-----------------------:|:----------------:|:---------------:|:---------------:|
| 1.5           | [Inference](#inference) \| [Training](#training) | N.A.            |   N.A.                 |  N.A.            |  [Inference](docs/en/controlnet.md) \| [Training](docs/en/controlnet.md) |    [Inference](../t2i_adapter/README.md#inference-and-examples)     |
| 2.0 & 2.1     | [Inference](#inference) \| [Training](#training) | [Inference](#image-variation) \| [Training](docs/en/image_variation_unclip.md)       |  [Inference](#inpainting)            | [Inference](#depth-to-image)     |   N.A.          |  [Inference](../t2i_adapter/README.md#inference-and-examples) \| [Training](../t2i_adapter/README.md#training)     |
| wukong       | [Inference](#inference) \| [Training](#training) | N.A.            |   [Inference](#inpainting)                |  N.A.            |  N.A. |    N.A.     |

> Although some combinations are not supported currently (due to the lack of checkpoints pretrained on the specific task and SD model), you can use the [Model Conversion](#model-conversion) tool to convert the checkpoint (e.g. from HF) then adapt it to the existing pipelines (e.g. image variation pipeline with SD 1.5)

You may click the link in the table to access the running instructions directly.

For model performance, please refer to [benchmark](benchmark.md).

## Installation

### Supported Platforms & Versions

Our code is mainly developed and tested on Ascend 910 platforms with MindSpore framework.
The compatible framework versions that are well-tested are listed as follows.

<div align="center">

| Ascend    |  MindSpore   | CANN   | driver | Python | MindONE |
|:-----------:|:----------------:|:--------:|:---------:|:------:|:---------:|
| 910      |     2.0         |   6.3 RC1   |  23.0.rc1 | 3.7.16  | master (4c33849)  |
| 910      |     2.1         |   6.3 RC2   |  23.0.rc2 | 3.9.18  | master (4c33849)  |
| 910*      |     2.2.1 (20231124)    |   7.1  | 23.0.rc3.6   |  3.7.16  | master (4c33849)  |

</div>

<!---
TODO: list more tested versions
-->

For detailed instructions to install CANN and MindSpore, please refer to the official webpage [MindSpore Installation](https://www.mindspore.cn/install).

**Note:** Running on other platforms (such as GPUs) and MindSpore versions may not be reliable.
It's highly recommended to use the verified CANN and MindSpore versions. More compatible versions will be continuously updated.

<details close markdown>

### Dependency

```shell
pip install -r requirements.txt
```

### Install from Source

```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone/examples/stable_diffusion_v2
```

</details>

## Dataset Preparation

<details close markdown>

This section describes the data format and protocol for diffusion model training.

The text-image pair dataset should be organized as follows.

```text
data_path
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

, where `img_txt.csv` is the image-caption file annotated in the following format.

```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

The first column is the image path related to the `data_path` and  the second column is the corresponding prompt.

For convenience, we have prepared two public text-image datasets obeying the above format.

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.
- [Chinese-art blip caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 100 chinese art-style images with BLIP-generated captions.

To use them, please download `pokemon_blip.zip` and `chinese_art_blip.zip` from the [openi dataset website](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets). Then unzip them on your local directory, e.g. `./datasets/pokemon_blip`.

</details>

## Text-to-Image

### Inference

#### Preparing Pretrained Weights
To generate images by providing a text prompt, please download one of the following checkpoints and put it in `models` folder:

<div align="center">

| **SD Version**     |  Lang.   | **MindSpore Checkpoint**                                                                                                          | **Ref. Official Model**                                                                           | **Resolution** |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------|
| 1.5                |   EN   | [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt)                    | [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)                    | 512x512        |
| 1.5-wukong         |   CN    | [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt)                 |          N.A.                                                                                     | 512x512        |
| 2.0                |   EN   |  [sd_v2_base-57526ee4.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt)              | [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)             | 512x512        |
| 2.0-v              |   EN   |  [sd_v2_768_v-e12e3a9b.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt)            | [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)                       | 768x768        |
| 2.1                |   EN   | [sd_v2-1_base-7c8d09ce.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_base-7c8d09ce.ckpt)          | [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)         | 512x512        |
| 2.1-v              |   EN   | [sd_v2-1_768_v-061732d1.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2-1_768_v-061732d1.ckpt)        | [stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)                   | 768x768        |

</div>

Take SD 1.5 for example:
```
cd examples/stable_diffusion_v2
wget https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt -P models
```

#### Text-to-Image Generation
After preparing the pretrained weight, you can run text-to-image generation by:

```shell
python text_to_image.py --prompt {text prompt} -v {model version}
```
> `-v`: model version. Valid values can be referred to `SD Version` in the above table.

For more argument illustration, please run `python text_to_image.py -h`.

Take SD 1.5 as an example:

```shell
# Generate images with the provided prompt using SD 1.5
python text_to_image.py --prompt "elven forest" -v 1.5
```

Take SD 2.0 as an example:
```shell
# Use SD 2.0 instead and add negative prompt guidance to eliminate artifacts
python text_to_image.py --prompt "elven forest" -v 2.0 --negative_prompt "moss" --scale 9.0 --seed 42
```

Here are some generation results.

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


### Training

Vanilla fine-tuning refers to training the whole UNet while freezing the CLIP-TextEncoder and VAE modules in the SD model.

To run vanilla fine-tuning, we will use the `train_text_to_image.py` script following the instructions below.

1. Prepare the pretrained checkpoint referring to [pretrained weights](#prepare-pretrained-weights)

2. Prepare the training dataset referring to [Dataset Preparation](#dataset-preparation).

3. Select a training configuration template from `config/train` and specify the `--train_config` argument. The selected config file should match the pretrained weight.
    - For SD1.5, use `configs/train/train_config_vanilla_v1.yaml`
    - For SD2.0 or SD2.1, use `configs/train/train_config_vanilla_v2.yaml`
    - For SD2.x with v-prediction, use `configs/train/train_config_vanilla_v2_vpred.yaml`

    Note that the model architecture (defined via `model_config`) and training recipes are preset in the yaml file. You may edit the file
     to adjust hyper-parameters like learning rate, training epochs, and batch size for your task.

4. Launch the training script after specifying the `data_path`, `pretrained_model_path`, and `train_config` arguments.

    ```shell
    python train_text_to_image.py \
        --train_config {path to pre-defined training config yaml} \
        --data_path {path to training data directory} \
        --output_path {path to output directory} \
        --pretrained_model_path {path to pretrained checkpoint file}
    ```
    > Please enable INFNAN mode by `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` for Ascend 910* if overflow found.

    Take fine-tuning SD1.5 on the Pokemon dataset as an example:

    ```shell
    python train_text_to_image.py \
        --train_config "configs/train/train_config_vanilla_v1.yaml" \
        --data_path "datasets/pokemon_blip/train" \
        --output_path "output/finetune_pokemon/txt2img" \
        --pretrained_model_path "models/sd_v1.5-d0ab7146.ckpt"
    ```

The trained checkpoints will be saved in {output_path}.

For more argument illustration, please run `python train_text_to_image.py -h`.

#### Distributed Training

For parallel training on multiple Ascend NPUs, please refer to the instructions below.

1. Generate the rank table file for the target Ascend server.

    ```shell
    python tools/hccl_tools/hccl_tools.py --device_num="[0,8)"
    ```
    > `--device_num` specifies which cards to train on, e.g. "[4,8)"

    A json file e.g. `hccl_8p_10234567_127.0.0.1.json` will be generated in the current directory after running.

2. Edit the distributed training script `scripts/run_train_distributed.sh` to specify
    1. `rank_table_file` with the path to the rank table file generated in step 1,
    2. `data_path`, `pretrained_model_path`, and `train_config` according to your task.

3. Launch the distributed training script by

    ```shell
    bash scripts/run_train_distributed.sh
    ```
    > Please enable INFNAN mode by `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` for Ascend 910* if overflow found.

    After launched, the training process can be traced by running `tail -f ouputs/train_txt2img/rank_0/train.log`.

    The trained checkpoints will be saved in `ouputs/train_txt2img`.

**Note:** For distributed training on large-scale datasets such as LAION, please refer to [LAION Dataset Preparation](tools/data_utils/README.md).


### LoRA Fine-tuning

Low-Rank Adaptation (LoRA) is a parameter-efficient finetuning method for large models.

Please refer to the tutorial of [LoRA for Stable Diffusion Finetuning](docs/en/lora_finetune.md) for detailed instructions.


### Dreambooth Fine-tuning

DreamBooth allows users to generate contextualized images of one subject using just 3-5 images of the subject, e.g., your dog.

Please refer to the tutorial of [DreamBooth for Stable Diffusion Finetuning](docs/en/dreambooth_finetune.md) for detailed instructions.


### Textual Inversion Fine-tuning

Coming soon

## Image-to-Image

### Image Variation

This pipeline uses a fine-tuned version of Stable Diffusion 2.1, which can be used to create image variations (image-to-image).
The pipeline comes with two pre-trained models, `2.1-unclip-l` and `2.1-unclip-h`, which use the pretrained CLIP Image embedder and OpenCLIP Image embedder separately.
You can use the `-v` argument to decide which model to use.
The amount of image variation can be controlled by the noise injected to the image embedding, which can be input by the `--noise_level` argument.
A value of 0 means no noise, while a value of 1000 means full noise.

#### Preparing Pretrained Weights
To generate variant images by providing a source image, please download one of the following checkpoints and put it in `models` folder:

<div align="center">

| **SD Version**     |  Lang.   | **MindSpore Checkpoint**                                                                                                          | **Ref. Official Model**                                                                 | **Resolution** |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------|
| 2.1-unclip-l       |   EN    | [sd21-unclip-l-baa7c8b5.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd21-unclip-l-baa7c8b5.ckpt)      | [stable-diffusion-2-1-unclip](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip)       | 768x768        |
| 2.1-unclip-h       |   EN    | [sd21-unclip-h-6a73eca5.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd21-unclip-h-6a73eca5.ckpt)       |   [stable-diffusion-2-1-unclip](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip)     | 768x768        |

</div>

And download the image encoder checkpoint [ViT-L-14_stats-b668e2ca.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/unclip/ViT-L-14_stats-b668e2ca.ckpt) to `models` folder.

#### Generating Image Variation

After preparing the pretrained weights, you can run image variation generation by:

```shell
python unclip_image_variation.py \
    -v {model version} \
    --image_path {path to input image} \
    --prompt "your magic prompt to run image variation."
```
> `-v`: model version. Valid values can be referred to `SD Version` in the above table.

For more argument usage, please run `python unclip_image_variation.py --help`

Using `2.1-unclip-l` model as an example, you may generate variant images based on the [example image](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png) by

```shell
python unclip_image_variation.py \
    -v 2.1-unclip-l \
    --image_path tarsila_do_amaral.png \
    --prompt "a cute cat sitting in the garden"
```

The output images will be saved in `output/samples` directory.

you can also add extra noise to the image embedding to increase the amount of variation in the generated images.

```shell
python unclip_image_variation.py -v 2.1-unclip-l --image_path tarsila_do_amaral.png --prompt "a cute cat sitting in the garden" --noise_level 200
```

<div align="center">
<img src="https://github.com/zhtmike/mindone/assets/8342575/393832cf-803a-4745-9fb1-7ef1107f9c37" width="760" />
</div>


For image-to-image fine-tuning, please refer to the tutorial of [Stable Diffusion unCLIP Finetuning](docs/en/image_variation_unclip.md) for detailed instructions.

### Inpainting

Text-guided image inpainting allows users to edit specific regions of an image by providing a mask and a text prompt, which is an interesting erase-and-replace editing operation.
When the prompt is set to empty, it can be applied to auto-fill the masked regions to fit the image context (which is similar to the AI fill and extend operations in PhotoShop-beta).

#### Preparing Pretrained Weights
To perform inpainting on an input image, please download one of the following checkpoints and put it in `models` folder:

<div align="center">

| **SD Version**     |  Lang.   | **MindSpore Checkpoint**                                                                                                          | **Ref. Official Model**                                                                 | **Resolution** |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------|
| 2.0-inpaint        |  EN      | [sd_v2_inpaint-f694d5cf.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_inpaint-f694d5cf.ckpt)        | [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | 512x512        |
| 1.5-wukong-inpaint |  CN      | [wukong-huahua-inpaint-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-inpaint-ms.ckpt) |                                                  N.A.                                               | 512x512        |
</div>

#### Running Image Inpainting

After preparing the pretrained weight, you can run image inpainting by:

```shell
python inpaint.py \
    -v {model version}
    --image {path to input image} \
    --mask  {path to mask image} \
    --prompt "your magic prompt to paint the masked region"
```
> `-v`: model version. Valid values can be referred to `SD Version` in the above table.

For more argument usage, please run `python inpaint.py --help`

Using `2.0-inpaint` as an example, you can download the [example image](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png) and [mask](https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png). Then execute

```shell
python inpaint.py \
    -v `2.0-inpaint`
    --image overture-creations-5sI6fQgYIuo.png \
    --mask overture-creations-5sI6fQgYIuo_mask.png \
    --prompt "Face of a yellow cat, high resolution, sitting on a park bench"
```

The output images will be saved in `output/samples` directory. Here are some generated results.

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/f0d6073e-fe24-4d3d-8f54-b7c4833bb206" width="100%" />
</div>
<p align="center">
<em> Text-guided image inpainting. From left to right: input image, mask, generated images. </em>
</p>

By setting empty prompt (`--prompt=""`), the masked part will be auto-filled to fit the context and background.
<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/21158de6-b9ec-4538-83cf-2a3bbea649e7" width="100%"
 />
</div>
<p align="center">
<em> Image inpainting. From left to right: input image, mask, generated images </em>
</p>



### Depth-to-Image

This pipeline allows you to generate new images conditioning on a depth map (preserving image structure) and a text prompt.
If you pass an initial image instead of a depth map, the pipeline will automatically extract the depth from it (using Midas depth estimation model)
and generate new images conditioning on the image depth, the image, and the text prompt.


#### Preparing Pretrained Weights

<div align="center">

| **SD Version**     |  Lang.   | **MindSpore Checkpoint**                                                                                                          | **Ref. Official Model**                                                                 | **Resolution** |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------|
| 2.0               |  EN       | [sd_v2_depth-186e18a0.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/stable_diffusion/sd_v2_depth-186e18a0.ckpt)        | [stable-diffusion-2-depth](https://huggingface.co/stabilityai/stable-diffusion-2-depth) | 512x512        |

</div>

And download the depth estimation checkpoint [midas_v3_dpt_large-c8fd1049.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/stable_diffusion/depth_estimator/midas_v3_dpt_large-c8fd1049.ckpt) to the `models/depth_estimator` directory.

#### Depth-to-Image Generation

After preparing the pretrained weight, you can run depth-to-image by:

```shell
# depth to image given a depth map and text prompt
python depth_to_image.py \
    --prompt {text prompt} \
    --depth_map {path to depth map} \
```

In case you don't have the depth map, you can input a source image instead, The pipeline will extract the depth map from the source image.

```shell
# depth to image conditioning on an input image and text prompt
python depth_to_image.py \
    --prompt {text prompt} \
    --image {path to initial image} \
    --strength 0.7
```
> `--strength` indicates how strong the pipeline will transform the initial image. A lower value - preserves more content of the input image. 1 - ignore the initial image and only condition on the depth and text prompt.

The output images will be saved in `output/samples` directory.

Example:

Download the [two-cat image](http://images.cocodataset.org/val2017/000000039769.jpg) and save it in the current folder. Then execute

```shell
python depth_to_image.py --image 000000039769.jpg --prompt "two tigers" --negative_prompt "bad, deformed, ugly, bad anatomy" \
```

Here are some generated results.

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/fa070832-d53f-4bd5-84af-ce8086f41866" width="100%"
 />
</div>
<p align="center">
<em> Text-guided depth-to-image. From left to right: input image, estimated depth map, generated images </em>
</p>

The two cats are replaced with two tigers while the background and image structure are mostly preserved in the generated images.


## ControlNet

ControlNet is a type of model for controllable image generation. It helps make image diffusion models more controllable by conditioning the model with an additional input image.
Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

For detailed instructions on inference and training with ControlNet, please refer to [Stable Diffusion with ControlNet](docs/en/controlnet.md).

## T2I Adapter

[T2I-Adapter](../t2i_adapter/README.md) is a simple and lightweight network that provides extra visual guidance for
Stable Diffusion models without re-training them. The adapter act as plug-in to SD models, making it easy to integrate
and use.

For detailed instructions on inference and training with T2I-Adapters, please refer to [T2I-Adapter](../t2i_adapter/README.md).

## Advanced Usage

### Model Conversion

We provide tools to convert SD 1.x or SD 2.x model weights from torch to MindSpore format. Please refer to [this doc](tools/model_conversion/README.md)

### Schedulers

Currently, we support the following diffusion schedulers.
- DDIM
- DPM Solver
- DPM Solver++
- PLMS
- UniPC

Detailed illustrations and comparison of these schedulers can be viewed in [Diffusion Process Schedulers](docs/en/schedulers.md).

### Training with v-prediction

The default objective function in SD training is to minimize the noise prediction error (noise-prediction).
To alter the objective to v-prediction, which is used in SD 2.0-v and SD 2.1-v, please refer to [v-prediction.md](docs/en/v_prediction.md)

### Diffusion Model Evaluation

We provide different evaluation methods including FID and CLIP-score to evaluate the quality of the generated images.
For detailed usage, please refer to [Evaluation for Diffusion Models](tools/eval/README.md)

### Safety Checker

Coming soon

### Watermark

Coming soon

## What's New
- 2023.12.01
  - Add ControlNet v1
  - Add unclip image variation pipeline, supporting both inference and training.
  - Add image inpainting pipeline
  - Add depth-to-image pipeline
  - Fix bugs and improve compatibility to support more Ascend chip types
  - Refractor documents
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
