# ControlNet based on Stable Diffusion
> [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

## Introduction
ControlNet controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

<p align="center">
   <img src="https://github.com/Gaohan123/mindone/assets/20148503/c5c27f00-3c20-479c-a540-70a0c8db0d48" width=500 />
</p>
<p align="center">
  <em> Figure 1. Illustration of a ControlNet [<a href="#references">1</a>] </em>
</p>



## Get Started
**MindONE** supports ControlNet generation for Stable Diffusion models based on MindSpore and Ascend platforms.

### Preparation

#### Dependency
- mindspore >= 2.0  [[install](https://www.mindspore.cn/install)]
- python >= 3.7

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

#### Trained Models
1. Canny edge maps:

   Please download the trained model [SD1.5-canny-ms checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_canny_sd_v1.5_static-6350d204.ckpt)

2. Segmentation edge maps:

   Please download the trained model [SD1.5-segmentation-ms checkpoint](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/control_segmentation_sd_v1.5_static-77bea2e9.ckpt)

3. Others:

   Coming soon.

Put them under `stable_diffusion_v2/models` folder.

#### Testing images preparation
Please prepare the images that you want to add extra conditions. There are some examples:

<div align="center">
<img src="https://github.com/Gaohan123/mindone/assets/20148503/24953d5f-dc20-45d4-ba45-ea602466eaa7" width="160" height="240" />
<img src="https://github.com/Gaohan123/mindone/assets/20148503/f1e21d57-7882-4e4f-a4c0-01568122e43b" width="160" height="240" />
</div>
<p align="center">
  <em> Images prepared to add extra controls </em>
</p>

Put them in an arbitrary directory on your machine. For example, `path/to/test_imgs`.

#### Load controls
1. Canny edge maps:
   It is implemented with opencv Canny API directly, you don't need to take more actions.

2. Segmentation edge maps (DeeplabV3Plus):
   The Segmentation Detector is implemented with [DeeplabV3Plus](https://arxiv.org/abs/1802.02611) [<a href="#references">2</a>]. Please download the pretrained model from [DeeplabV3Plus checkpoint](https://download.mindspore.cn/models/r1.9/deeplabv3plus_s16_ascend_v190_voc2012_research_cv_s16acc79.06_s16multiscale79.96_s16multiscaleflip80.12.ckpt)

   There is an example of segmentation edge map with an image of bird.

   <div align="center">
   <img src="https://github.com/Gaohan123/mindone/assets/20148503/dd4769f3-caaf-4dad-80df-5905ab6260d9" width="160" height="240" />
   </div>
   <p align="center">
     <em> segmentation edge map with an image of bird </em>
   </p>

   Attention: As the DeeplabV3Plus is trained on VOC dataset, currently it only supports prompts related to the objects: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'. More is coming soon.

### Parameter Setting

Go to the directory `stable_diffusion_v2/inference` first.

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

### Image Generation

   Go to the directory `stable_diffusion_v2/inference` first. Run the command below to generate images

```shell
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

#### Important arguments in the shell scripts
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


## Reference
[1] [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

[2] [Encoder-decoder with atrous separable convolution for semantic image segmentation](https://arxiv.org/abs/1802.02611)
