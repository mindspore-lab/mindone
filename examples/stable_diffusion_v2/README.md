# Stable Diffusion based on MindSpore

<p float="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/46a3b7f1-240d-4933-a2c7-f49ce995ffd5" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/91c5cff0-5a35-4624-98f4-61777fd883a0" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/37f3e414-a1a4-4743-8ad1-5de7c10c3036" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/9e456f4e-8174-467c-8d0c-7a42c9f19dc7" width="25%" />
</p>
<p float="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/86d63435-fb41-423f-8d11-d55185924fd7" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/b2dd903b-aafb-4552-a7fb-3da5f7ae8baa" width="25%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/e146e0ff-9b49-4cf6-8458-c2abea29cc6b" width="16.66%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/8994ca17-5d7c-41fe-bfe3-baece43baddc" width="16.66%" /><img src="https://github.com/SamitHuang/mindone/assets/8156835/fc9d2149-e3a3-4047-b343-fbdab354d3d0" width="16.66%" />
</p>

## Table of Contents
- [Introduction](#introduction)
    - [Supported Models and Pipelines](#supported-models-and-pipelines)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Text-to-Image](#text-to-image)🔥
    - [Inference](#inference)
    - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [LoRA Fine-tuning](#lora-fine-tuning) 🔥
    - [Dreambooth Fine-tuning](#dreambooth-fine-tuning)
    - [Textual Inversion Fine-tuning](#textual-inversion-fine-tuning)
    - [Benchmark](#benchmark)
- [Image-to-Image](#image-to-image)
    - [Inpainting](#inpainting)
    - [Depth-to-Image](#depth-to-image)
- [Advanced Usage](#advanced-usage)
    - [Model Conversion](#model-conversion)
    - [Schedulers](#schedulers)
    - [Training with v-prediction](#training-with-v-prediction)
    - [Diffusion Model Evaluation](#diffusion-model-evaluation)

## Introduction

This repository integrates state-of-the-art [Stable Diffusion](https://arxiv.org/abs/2112.10752) models including SD1.5, SD2.0, and SD2.1,
supporting various generation tasks and pipelines. Efficient training and fast inference are implemented based on MindSpore.

<!--
This repository provides "small" but popularly used diffusion models like SD1.5. Currently, we support the following tasks and models.
-->

### Supported Models and Pipelines

#### SD1.5
| **text-to-image** |
|:--------------:|
| [Inference](#inference) \| [Training](#training) |

#### SD2.0 & SD2.1
| **text-to-image**      | **inpainting**  | **depth-to-image**  |
|:--------------:|:--------------------:|:-----------------------:|
| [Inference](#inference) \| [Training](#training) |  [Inference](#inpainting)            | [Inference](#depth-to-image)     |

You may click the link in the table to access the running instructions directly.

## Installation

### Requirements

| ascend    |  mindspore   | cann   | driver | python |
|:-----------:|:----------------:|:--------:|:---------:|:------:|
| 910      |     2.1.0         |   6.3.RC2   |  24.1.RC1 | 3.9  |
| 910*      |     2.3.0     |   7.3  | 23.0.3   |  3.8  |
| 910*      |     2.3.1     |   8.0.RC2.bata1  | 24.1.RC2   |  3.8  |



For detailed instructions to install CANN and MindSpore, please refer to the official webpage [MindSpore Installation](https://www.mindspore.cn/install).

To install other dependent packages:

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

This section describes the data format and protocol for diffusion model training.

The text-image pair dataset should be organized as follows.

```text
data_path
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

where `img_txt.csv` is the image-caption file annotated in the following format.

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

| **sd version**     |  language   | **mindspore checkpoint**                                                                                                          | **ref. official model**                                                                           | **resolution** |
|:--------------------:|:----------:|:-------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:----------------:|
| 1.5                |   EN   | [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt)                    | [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)                    | 512x512        |
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

<div align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/1c35853d-036f-459c-944c-9953d2da8087" width="320" />
</div>


Take SD 2.0 as an example:
```shell
# Use SD 2.0 instead and add negative prompt guidance to eliminate artifacts
python text_to_image.py --prompt "elven forest" -v 2.0 --negative_prompt "moss" --scale 9.0 --seed 42
```

</div>
<p align="center">
<img src="https://github.com/SamitHuang/mindone/assets/8156835/b1f037ca-4e03-40e4-8da2-d358801eadd5)" width="320" />


##### Inference with different samplers
By default, the inference use dpm++ 2M samplers. You can use others if needed. The support list and detailed illustrations refer to  [schedulers](docs/en/schedulers.md).

##### Distributed Inference

  For parallel inference, take SD1.5 on the Chinese art dataset as an example:

   ```shell
   bash scripts/run_infer_distributed.sh  
   ```
   > Note: Parallel inference only can be used for mutilple-prompt.

##### Long Prompts Support

  By default, SD V2(1.5) only supports the token sequence no longer than 77. For those sequences longer than 77, they will be truncated to 77, which can cause information loss.

  To avoid information loss for long text prompts, we can divide one long tokens sequence (N>77) into several shorter sub-sequences (N<=77) to bypass the constraint of context length of the text encoders. This feature is supported by `args.support_long_prompts` in `text_to_image.py`.

  When running inference with `text_to_image.py`, you can set the arguments as below.

  ```bash
  python text_to_image.py \
  ...  \  # other arguments configurations
  --support_long_prompts True \  # allow long text prompts
  ```


##### Flash-Attention Support

  Flash attention supported by setting the argument `enable_flash_attention` as `True` in `configs/v1-inference.yaml` or `configs/v2-inference.yaml`. For example, in `configs/v1-inference.yaml`:

  ```
      unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        ...
        enable_flash_attention: False
        fa_max_head_dim: 256  # max head dim of flash attention. In case of oom, reduce it to 128
  ```
  One can set `enable_flash_attention` to `True`. In case of OOM (out of memory) error, please reduce the `fa_max_head_dim` to 128.




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

    ```shell
    bash scripts/run_train_distributed.sh
    ```

   After launched, the training process can be traced by running `tail -f ouputs/train_txt2img/worker_0.log`.

   The trained checkpoints will be saved in `ouputs/train_txt2img`.

**Note:** For distributed training on large-scale datasets such as LAION, please refer to [LAION Dataset Preparation](tools/data_utils/README.md).


### LoRA Fine-tuning

Low-Rank Adaptation (LoRA) is a parameter-efficient finetuning method for large models.

Please refer to the tutorial of [LoRA for Stable Diffusion Finetuning](docs/en/lora_finetune.md) for detailed instructions.


### Dreambooth Fine-tuning

DreamBooth allows users to generate contextualized images of one subject using just 3-5 images of the subject, e.g., your dog.

Please refer to the tutorial of [DreamBooth for Stable Diffusion Finetuning](docs/en/dreambooth_finetune.md) for detailed instructions.


### Textual Inversion Fine-tuning

Textual Inversion learns one or a few text embedding vectors for a new concept, e.g., object or style, with only 3~5 images.

Please refer to the tutorial of [Textual Inversion for Stable Diffusion Finetuning](docs/en/textual_inversion_finetune.md) for detailed instructions.


### Benchmark
For model performance, please refer to [benchmark](benchmark.md).

## Image-to-Image

#### Generating Image Variation

### Inpainting

Text-guided image inpainting allows users to edit specific regions of an image by providing a mask and a text prompt, which is an interesting erase-and-replace editing operation.
When the prompt is set to empty, it can be applied to auto-fill the masked regions to fit the image context (which is similar to the AI fill and extend operations in PhotoShop-beta).

#### Preparing Pretrained Weights
To perform inpainting on an input image, please download one of the following checkpoints and put it in `models` folder:

<div align="center">

| **sd version**     |  language   | **mindspore checkpoint**                                                                                                          | **ref. official model**                                                                           | **resolution** |
|:--------------------:|:----------:|:-------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:----------------:|
| 2.0-inpaint        |  EN      | [sd_v2_inpaint-f694d5cf.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_inpaint-f694d5cf.ckpt)        | [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | 512x512        |
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
    -v "2.0-inpaint" \
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

| **sd version**     |  language   | **mindspore checkpoint**                                                                                                          | **ref. Official model**                                                                           | **resolution** |
|:--------------------:|:----------:|:-------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------:|:----------------:|
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

### Inference on pre-trained models derived from SD

You could infer with other existing pre-trained models derived from SD, which has undergone extensive fine-tuning processes or trained from scratch on specific datasets. Convert the weights from torch to mindsport format first and do inference with samplers.

Here we provide an example of running inference on the Deliberate Model. Please refer to the instructions here, [Inference with the Deliberate Model](docs/en/inference_with_deliberate.md).

### Training with v-prediction

The default objective function in SD training is to minimize the noise prediction error (noise-prediction).
To alter the objective to v-prediction, which is used in SD 2.0-v and SD 2.1-v, please refer to [v-prediction.md](docs/en/v_prediction.md)

### Diffusion Model Evaluation

We provide different evaluation methods including FID and CLIP-score to evaluate the quality of the generated images.
For detailed usage, please refer to [Evaluation for Diffusion Models](tools/eval/README.md)

## What's New
- 2024.01.10
  - Add Textual Inversion fine-tuning
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
