# ControlNet based on Stable Diffusion
> [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

# Table of Content

1. [Introduction](#introduction)
2. [Inference](#inference)
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
|   SD1.5            |  EN      | [SD1.5-segmentation-ms checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_segmentation_sd_v1.5_static-77bea2e9.ckpt) |       [control_sd15_seg.pth](https://huggingface.co/lllyasviel/ControlNet/tree/main/models)                                                                                      | 512x512        |
|   SD1.5            |  EN      | [SD1.5-openpose-ms checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_openpose_sd_v1.5_static-6167c529.ckpt)        | [control_sd15_openpose.pth](https://huggingface.co/lllyasviel/ControlNet/tree/main/models) | 512x512        |

### Preparing Control Signals

Please prepare the source images that you want to extract the control signals from (e.g. canny edge, segmentation map, openpose).

Here are three examples:

<div align="center">
<img src="https://github.com/Gaohan123/mindone/assets/20148503/24953d5f-dc20-45d4-ba45-ea602466eaa7" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/f1e21d57-7882-4e4f-a4c0-01568122e43b" width="160" height="240" />
<img src="https://github.com/congw729/mindone/assets/115451386/fd1180ad-878c-44f4-988c-b18c56972015" width="160" height="240" />
</div>
<p align="center">
  <em> Images prepared to add extra controls </em>
</p>

You may download and save them under `test_images/` folder.

- For edge control, the canny edge is extracted using opencv Canny API in the inference script. There is no need to extract it manually.

- For segmentation map control, you can download the follow segmentation map extracted using [DeeplabV3Plus](https://arxiv.org/abs/1802.02611), download [model weights](https://download.mindspore.cn/models/r1.9/deeplabv3plus_s16_ascend_v190_voc2012_research_cv_s16acc79.06_s16multiscale79.96_s16multiscaleflip80.12.ckpt)

   Attention: As the DeeplabV3Plus is trained on VOC dataset, currently it only supports prompts related to the objects: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'. More is coming soon.

- For openpose control, you need download the openpose models' weights for [body pose](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/ms_body_pose_model.ckpt) extraction and [hand pose](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/ms_hand_pose_model.ckpt) extraction, then put them under `models/` folder.


<div align="center">
<img src="https://github.com/congw729/mindone/assets/115451386/74a23fc0-2b68-4f8e-88ff-fb3cf9770e16" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/dd4769f3-caaf-4dad-80df-5905ab6260d9" width="160" height="240" />
<img src="https://github.com/congw729/mindone/assets/115451386/2a940947-52b0-4a0d-a5b8-7516a35fc5ca" width="160" height="240" />
</div>
<p align="center">
  <em> Control maps generated by canny/segmentation/openpose detectors </em>
</p>


### Setting Arguments

Before running the inference script, please set up the arguments as follows.

1. Canny edge:

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
3. Openpose:

   Open the file `stable_diffusion_v2/inference/config/controlnet_openpose.yaml`. Set arguments as below:
   ```yaml
   image_path: "test_imgs/pose1.png" # Image to inpaint
   prompt: "Chief in the kitchen" # text prompt
   ```

   Open the file `stable_diffusion_v2/inference/config/model/v1-inference-controlnet.yaml`. Set argument as below:
   ```yaml
   pretrained_ckpt: "stable_diffusion_v2/models/control_openpose_sd_v1.5_static-6167c529.ckpt" # pretrained controlnet model weights with openpose
   condition_ckpt_path: "folder_path/to/openpose_ckpt" # If not set, will use ../models by default
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

> For openpose control, please set `--controlnet_mode` with "openpose".

Key arguments:
- `device_target`: Device target, should be in [Ascend, GPU, CPU]. (Default is `Ascend`)
- `task`: Task name, should be [text2img, img2img, inpaint, controlnet], if choose a task name, use the config/[task].yaml for inputs.
- `model`: Path to config which constructs model.
- `sampler`: Infer sampler yaml path.
- `sampling_steps`: Number of sampling steps.
- `n_iter`: Number of iterations or trials.
- `n_samples`: How many samples to produce for each given prompt in an iteration. A.k.a. batch size.
- `controlnet_mode`: Control mode for controlnet, should be in [canny, segmentation, openpose]

### Results
Generated images will be saved in `stable_diffusion_v2/inference/output/samples` by default.
Here are some examples:

<div align="center">
<img src="https://github.com/congw729/mindone/assets/115451386/d7059d84-7740-4260-9a1f-9149a32bdfb1" width="160" height="240" />
<img src="https://github.com/congw729/mindone/assets/115451386/27653ee7-4bae-498f-b01b-45d76ee9d8a3" width="160" height="240" />
<img src="https://github.com/congw729/mindone/assets/115451386/d5dafa64-779d-4247-bc04-530c07594a52" width="160" height="240" />
</div>
<p align="center">
  <em> Generated Images with ControlNet on canny control maps </em>
</p>

<div align="center">
<img src="https://github.com/Gaohan123/mindone/assets/20148503/6d543d0b-e1c2-447b-805a-19d9253a488b" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/90835ad9-38aa-4ca2-862a-0344c0760463" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/bf1bc4e9-c16c-4d37-8b72-cbc83fd8569e" width="160" height="240" />
</div>
<p align="center">
  <em> Generated Images with ControlNet on DeeplabV3Plus segmentation edge maps </em>
</p>

<div align="center">
<img src="https://github.com/congw729/mindone/assets/115451386/7ec0ad84-5120-4475-979e-b8732f7d187a" width="160" height="240" />
<img src="https://github.com/congw729/mindone/assets/115451386/5b227858-d799-4cbb-bae7-d9956e79d6f9" width="160" height="240" />
<img src="https://github.com/congw729/mindone/assets/115451386/a25a69d9-42ba-4e41-bf77-4e1761370d8b" width="160" height="240" />
</div>

<p align="center">
  <em> Generated Images with ControlNet on openpose control maps </em>
</p>

## Fine-tuning using ControlNet

Try to customize your own ControlNet? Downloading the SD base weight, preparing your dataset and start training!

Now only support training based on SD1.5.ControlNet

All codes have been tested on Ascend 910* with MindSpore 2.2 20231124 version.

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
> Please enable INFNAN mode by `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` for Ascend 910* if overflow found.

The resulting log will be saved in $output_dir as defined in the script, and the saved checkpoint will be saved in $output_path as defined in  `train_config` file.


#### 4. Evaluation
To evaluate the training result, please modify the control image path in the script and indicate the path to the trained checkpoint, and run the following script.

```
sh scripts/run_infer_cldm.sh $CARD_ID $CHECKPOINT_PATH $OUTPUT_FOLDER_NAME
```
The result would be saved at ./inference/output/$OUTPUT_FOLDER_NAME.

Here are some inference results after training Fill50k dataset (lr=5e-4, bs=2, epoch=4):

![controlnet_train_validate](https://github.com/congw729/mindone/assets/115451386/62ad40a3-0510-4606-84cb-780c72989b36)

Comparing the results with ground truth:
![controlnet_train_gt](https://github.com/congw729/mindone/assets/115451386/644e436c-4ac9-4b23-81ff-6c0f0dbef9f8)


Control:
![controlnet_train_control](https://github.com/congw729/mindone/assets/115451386/8b695f01-4220-4d4f-afff-01ae604169e0)

Prompt:
![controlnet_train_prompt](https://github.com/congw729/mindone/assets/115451386/7adb909a-a4b3-4ca3-856a-b996233b7b33)


## Reference
[1] [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

[2] [Encoder-decoder with atrous separable convolution for semantic image segmentation](https://arxiv.org/abs/1802.02611)
