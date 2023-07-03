# Stable Diffusion

This folder contains Stable Diffusion models implemented with MindSpore. It targets full support for inference, finetuning, and training from scratch. New checkpoints and features will be continuously updated

## Features
- [x] SD1.4 inference, support Chinese prompt. Though runnable with English prompts, the generation quality is much worse than CN prompts.
- [x] SD1.4 finetune, support Chinese image-text pair data.
- [x] SD2.0 inference, support English prompt. It does not support Chinese prompts.
- [x] SD2.0 finetune, support English image-text par data.

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


- - -
## Stable Diffusion 2.0 - EN
### Inference

Please first prepare the pre-trained SD model. The default path is `./models`.

Then, inference SD with different samplers:

```bash
# Text to image generation with SD2.0, the default sampler is PLMS
python text_to_image.py --prompt "A wolf in winter"
# Text to image generation with SD2.0, using PLMS sampler
bash scripts/tests/test_plms_sampler.py
# Text to image generation with SD2.0, using DDIM sampler
bash scripts/tests/test_ddim_sampler.py
# Text to image generation with SD2.0, using DPM-Solver sampler
bash scripts/tests/test_dpmsolver_sampler.py
# Text to image generation with SD2.0, using DPM-Solver++ sampler
bash scripts/tests/test_dpmsolverpp_sampler.py
# Text to image generation with SD2.0, using UniPC sampler
bash scripts/tests/test_unipc_sampler.py
```

For more argument usages, please run `python text_to_image.py -h`.

Some text-to-image generation examples are shown here:

```bash
A Van Gogh style oil painting of sunflower
```

| PLMS | DDIM | DPM-Solver | DPM-Solver++ | UniPC |
| :----: | :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/1105da61-4f12-47d3-a008-25117fddfe68" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ba5f89e8-84a6-4805-a132-34d0aff4f91a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/7adf2a87-a1ed-4963-8c00-4d70e34c820c" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/4cfed3e7-1dff-49f1-8399-e25593d29e83" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e7d9e51f-50f8-4ed6-9685-431b813967d1" width="155" height="155" /> |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9f1a5530-87ac-4fa4-adc2-3b304bfc636d" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/430cc134-16cb-4327-9b88-1bc6de99f33b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2ae82e37-f27a-4805-8d05-71c8a8f8676e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b71626a5-2d39-4c70-aee7-e68cc2c10651" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b693bcaa-479c-4fdf-adce-22afd453f975" width="155" height="155" /> |

```bash
A photo of an astronaut riding a horse on mars
```

| PLMS #1 | PLMS #2 | PLMS #3 | PLMS #4 |
| :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/9c80d7fe-4709-4387-b51d-fe9b86d1e92a" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/a66c1c3d-4c81-4c21-8714-3afe28769122" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/ae6af084-7930-42fd-a91d-7aaf182f5f5b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/03384b6d-21ba-49ec-af6b-c029a2ff8e37" width="155" height="155" /> |

```bash
A high tech solarpunk utopia in the Amazon rainforest
```

| PLMS #1 | PLMS #2 | PLMS #3 | PLMS #4 |
| :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/e1eeef11-0aeb-43f7-8b40-8e2a0c0e9a70" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2faa35c9-c52b-4753-afdc-ea3b24afb2d2" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/fcb0a813-1bfc-4d3e-a6f3-eb4359fee72b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/5a5199f8-6a15-4930-889e-5876e25b01cb" width="155" height="155" /> |

```bash
The beautiful night view of the city has various buildings, traffic flow, and lights
```

| PLMS #1 | PLMS #2 | PLMS #3 | PLMS #4 |
| :----: | :----: | :----: | :----: |
| <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/71658f30-d89d-4e34-9195-34e14a132d3b" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/b3fdcf9b-699d-4717-a997-c7f7fac4858e" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/c1673c78-ec80-46aa-86b0-e087207ae390" width="155" height="155" /> | <img src="https://github.com/zhaoyuzhi/mindone/assets/13333802/2e099b94-99f6-4488-a625-58cbf1fce179" width="155" height="155" /> |

### Vanilla Finetuning

Vanilla finetuning refers to the second-stage training in the LDM paper. Only the latent diffusion model (**UNet** + ddpm) will be trained and updated, while CLIP and AutoEncoder are frozen.  

```shell
sh scripts/run_train_v2.sh
```

Modify `data_path` in `run_train_v2.sh` to the path to the dataset that you want to train on. 

### LoRA Finetuning

LoRA finetuning has lower memory requirement and allows finetuning on images with higher-resolution such as 768x768.

Coming soon.

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

- 2023.06.12  Add velocity parameterization for DDPM prediction type. Usage: set `parameterization: velocity` in configs/your_train.yaml  


## TODO
- [ ] Support vanilla finetuning a.k.a second-stage training for 768x768 images. (Currently, it will lead to OOM on a single card)
- [ ] Fix bugs in loading pretrained checkpoints (some network params are not in the pretrained checkpoint)
- [ ] Support SD2.1 (768) inference and finetuning.
- [ ] Support training from scratch including first-stage training

