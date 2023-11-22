# ControlNet based on Stable Diffusion
> [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

# Table of Content

1. [Introduction](#introduction)
2. [Get Started](#get-started)
3. [Fine-tuning using ControlNet](#fine-tuning-using-controlnet)

## Introduction
ControlNet controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

<p align="center">
   <img src="https://github.com/Gaohan123/mindone/assets/20148503/c5c27f00-3c20-479c-a540-70a0c8db0d48" width=500 />
</p>
<p align="center">
  <em> Figure 1. Illustration of a ControlNet [<a href="#references">1</a>] </em>
</p>

## Dependency

Please refer to the [Installation](../../README.md#installation) section.


## Inference

### Preparing Pretrained Weights

To perform controllable image generation with existing ControlNet checkpoints, please download one of the following checkpoints and put it in `models` folder:

| **SD Version**     |  Lang.   | **MindSpore Checkpoint**                                                                                                          | **Ref. Official Model**                                                                 | **Resolution** |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|----------------|
|   SD1.5            |  EN      | [SD1.5-canny-ms checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_canny_sd_v1.5_static-6350d204.ckpt)        | [control_sd15_canny.pth](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) | 512x512        |
|   SD1.5            |  EN      | [SD1.5-segmentation-ms checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_segmentation_sd_v1.5_static-77bea2e9.ckpt) |       [control_sd15_seg.pth](https://huggingface.co/lllyasviel/ControlNet/tree/main/models)                                    N.A.                                                      | 512x512        |


### Preparing Control Signals

Please prepare the source images that you want to extract the control signals from (e.g. canny edge, segmentation map).

Here are two examples:

<div align="center">
<img src="https://github.com/Gaohan123/mindone/assets/20148503/24953d5f-dc20-45d4-ba45-ea602466eaa7" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/f1e21d57-7882-4e4f-a4c0-01568122e43b" width="160" height="240" />
</div>
<p align="center">
  <em> Images prepared to add extra controls </em>
</p>

You may download and save them in `test_images/.`.

- For edge control, the canny edge is extracted using opencv Canny API in the inference script. There is no need to extract it manually.

- For segmentation map control, you can download the follow segmentation map extracted using [DeeplabV3Plus](https://arxiv.org/abs/1802.02611).

   <div align="center">
   <img src="https://github.com/Gaohan123/mindone/assets/20148503/dd4769f3-caaf-4dad-80df-5905ab6260d9" width="160" height="240" />
   </div>
   <p align="center">
     <em> segmentation edge map with an image of bird </em>
   </p>

   Attention: As the DeeplabV3Plus is trained on VOC dataset, currently it only supports prompts related to the objects: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'. More is coming soon.


### Setting Arguments

Before running the inference script, please set up the arguments as follows.

1. Canny edge maps:

   Open the file `stable_diffusion_v2/inference/config/controlnet_canny.yaml`. Set arguments as below:
   ```yaml
   image_path: "test_imgs/dog2.png" # Image to inpaint
   prompt: "cute dog" # text prompt
   ```

   Open the file `stable_diffusion_v2/inference/config/model/v1-inference-controlnet.yaml`. Set argument as below:
   ```yaml
   pretrained_ckpt: "stable_diffusion_v2/models/control_canny_sd_v1.5_static-6350d204.ckpt" # pretrained controlnet model weights with canny edge
   ```

2. Segmentation edge maps (DeeplabV3Plus):

   Open the file `stable_diffusion_v2/inference/config/controlnet_segmentation.yaml`. Set arguments as below:
   ```yaml
   image_path: "test_imgs/bird.png" # Image to inpaint
   prompt: "Bird" # text prompt
   condition_ckpt_path: "models/deeplabv3plus_s16_ascend_v190_voc2012_research_cv_s16acc79.06_s16multiscale79.96_s16multiscaleflip80.12.ckpt" # segmentation control model
   ```

   Open the file `stable_diffusion_v2/inference/config/model/v1-inference-controlnet.yaml`. Set argument as below:
   ```yaml
   pretrained_ckpt: "stable_diffusion_v2/models/control_segmentation_sd_v1.5_static-77bea2e9.ckpt" # pretrained controlnet model weights with segmentation
   ```

### Generating Images with ControlNet

After preparing the checkpoints and setting up the arguments, you run ControlNet inference as follows.

```shell
cd stable_diffusion_v2/inference

python sd_infer.py \
--device_target=Ascend \
--task=controlnet \
--model=./config/model/v1-inference-controlnet.yaml \
--sampler=./config/schedule/ddim.yaml \
--sampling_steps=20 \
--n_iter=1 \
--n_samples=4 \
--controlnet_mode=canny
```
> For segmentation control, please set `--controlnet_mode` with "segmentation".

Key arguments:
- `device_target`: Device target, should be in [Ascend, GPU, CPU]. (Default is `Ascend`)
- `task`: Task name, should be [text2img, img2img, inpaint, controlnet], if choose a task name, use the config/[task].yaml for inputs.
- `model`: Path to config which constructs model.
- `sampler`: Infer sampler yaml path.
- `sampling_steps`: Number of sampling steps.
- `n_iter`: Number of iterations or trials.
- `n_samples`: How many samples to produce for each given prompt in an iteration. A.k.a. batch size.
- `controlnet_mode`: Control mode for controlnet, should be in [canny, segmentation]

### Results
Generated images will be saved in `stable_diffusion_v2/inference/output/samples` by default.
Here are samples generated by a bird image with DeeplabV3Plus segmentation edge maps:

<div align="center">
<img src="https://github.com/Gaohan123/mindone/assets/20148503/6d543d0b-e1c2-447b-805a-19d9253a488b" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/90835ad9-38aa-4ca2-862a-0344c0760463" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/bf1bc4e9-c16c-4d37-8b72-cbc83fd8569e" width="160" height="240" />
</div>
<p align="center">
  <em> Generated Images with ControlNet on DeeplabV3Plus segmentation edge maps </em>
</p>

## Fine-tuning using ControlNet

Try to customize your own ControlNet? Downloading the SD base weight, preparing your dataset and start training!

Now only support training based on SD1.5.ControlNet

### Environment

- Hardware: 910B
- MindSpore: 2.2 20231114

### Train a ControlNet from SD1.5

#### 1. Model Weight Conversion

Once the [stable diffusion v1.5 model weights](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) (for mindspore)  are saved in `models`, you can run the following command to create a ControlNet init checkpoint.

```
python tools/sd_add_control.py
```

#### 2. Data Preparation

We will use [Fill50k dataset](https://openi.pcl.ac.cn/attachments/5208caad-1727-46cc-b34e-add9afbd0557?type=1) to let the model learn to generate images following the edge control. Download it and put it under `datasets/` folder

For convenience, we take the first 1K samples as training set, which can done by keeping the first 1000 lines in `prompt.json` and removing the rest.

If you want to use your own dataset, please make sure the structure as belows:
```text
dir
    ├── prompt.json
    ├── source
    └── target
```

`source` and `target` is the file folder with all images. The difference is the images under `target` folder are original image or called target image, and the images under `source` are the canny/segementation/other control images generated from original images. (eg.for Fill50k dataset `source/img0.png` is the canny image of `target/img0.png` )

```text
dir
├── img1.png
├── img2.png
├── img3.png
└── ...
```

`prompt.json` is the annotation file in the following format

```json
{"source": "source/0.png", "target": "target/0.png", "prompt": "pale golden rod circle with old lace background"}
{"source": "source/1.png", "target": "target/1.png", "prompt": "light coral circle with white background"}
{"source": "source/2.png", "target": "target/2.png", "prompt": "aqua circle with light pink background"}
{"source": "source/3.png", "target": "target/3.png", "prompt": "cornflower blue circle with light golden rod yellow background"}
```
#### 3. Training


We will use the `scripts/run_train_cldm.sh` script for finetuning. Before running, please make sure the arguments `data_path` and `pretrained_model_path` are set to your own path, for example

```shell
data_path=/path_to_dataset
pretrained_model_path=/path_to_init_model
```

Then, check the training settings in `train_config`, some params default setting are as belows:

```text
train_batch_size: 2
optim: "adamw"
start_learning_rate: 5.e-4
```

For more settings, check the `model_config` file.

The default `sd_locked` is True, which means only update the parameters of ControlNet model, not the decoder part of Unet model.
The default `only_mid_control` is False, which means the data processed by ControlNet will go through the decoder part of Unet model.

For more explanation, please see [ControlNet official](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md#other-options)

Final, execute the script to launch finetuning

```
sh scripts/run_train_cldm.sh $CARD_ID
```

The resulting log will be saved in $output_dir as defined in the script, and the saved checkpoint will be saved in $output_path as defined in  `train_config` file.


#### 4. Evaluation
To evaluate the training result, please run the following script and indicate the path to the trained checkpoint.

```
sh scripts/run_infer_cldm.sh $CARD_ID $CHECKPOINT_PATH
```

And modify the control image path in the script.



## Reference
[1] [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

[2] [Encoder-decoder with atrous separable convolution for semantic image segmentation](https://arxiv.org/abs/1802.02611)
