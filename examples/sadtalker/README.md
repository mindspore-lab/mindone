# SadTalker :sob:

A Mindspore implementation of [SadTalker](https://arxiv.org/pdf/2211.12194.pdf)  based on its original implemetation.

## Introduction
SadTalker is a novel system for a stylized audio-driven single image talking head videos animation using the generated realistic 3D motion coefficients (head pose, expression) of the 3DMM.

<p align="center">
<img src="https://github.com/hqkate/sadtalker/assets/26082447/d9d3b2d5-1e80-4304-84b4-768ce2b9814c" title="SadTalke" width="50%"/>

<br>
<b>TL;DR: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; single portrait image 🙎‍♂️  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; audio 🎤  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; =  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; talking head video 🎞.</b>
<br>

</p>


## Installation

```bash
pip install -r requirements.txt
```

## Data and Preparation

To execute the inference pipeline of SadTalker, please first download the [pretrained checkpoints](#pretrained-checkpoints) and setup the path in [config file](./config/sadtalker-infer.yaml).

### Pretrained checkpoints

You can download the checkpoints from this [link](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/).
The description of each model is as follows:

**1. Model Network Pretrained weights**

| Model | Description | Required-by | MindSpore Checkpoint |
| --- | --- | --- | --- |
| ms_audio2exp.ckpt  | Pre-trained ExpNet in Sadtalker. | Infer / Finetune | [ms_audio2exp.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_audio2exp.ckpt)
| ms_audio2pose.ckpt | Pre-trained PoseVAE in Sadtalker. | Infer / Finetune | [ms_audio2pose.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_audio2pose.ckpt) |
| ms_mapping.ckpt | Pre-trained MappingNet for cropped image in Sadtalker. | Infer / Finetune | [ms_mapping.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_mapping.ckpt) |
| ms_mapping_full.ckpt | Pre-trained MappingNet for full image in Sadtalker. | Infer / Finetune | [ms_mapping_full.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_mapping_full.ckpt) |
| ms_kp_extractor.ckpt | Pre-trained KPDetector in face-vid2vid model from [the reappearance of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). | Infer / Finetune | [ms_kp_extractor.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_kp_extractor.ckpt) |
| ms_he_estimator.ckpt | Pre-trained HEEstimator in face-vid2vid model from [the reappearance of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). | Infer / Finetune | [ms_he_estimator.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_he_estimator.ckpt) |
| ms_generator.ckpt | Pre-trained OcclusionAwareSPADEGenerator in face-vid2vid model from [the reappearance of face-vid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis). | Infer / Finetune | [ms_generator.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_generator.ckpt) |
| ms_net_recon.ckpt | Pre-trained 3DMM extractor in [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction). | Infer / Finetune | [ms_net_recon.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_net_recon.ckpt) |
| ms_wav2lip.ckpt | Pre-trained highly accurate lip-sync model in [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Train / Finetune | [ms_wav2lip.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_wav2lip.ckpt) |
| ms_vgg19.ckpt | Pre-trained VGG19 model | Train / Finetune | [ms_vgg19.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_vgg19.ckpt) |
| ms_hopenet_robust_alpha1.ckpt | Pre-trained head pose estimation model in [Hopenet](https://github.com/natanielruiz/deep-head-pose/tree/master) | Train / Finetune | [ms_hopenet_robust_alpha1.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/ms/ms_hopenet_robust_alpha1.ckpt) |

**2. 3DMM library files**

| Model | Description | Required-by | Download link |
| --- | --- | --- | --- |
| BFM_Fitting/ | 3DMM library files. | Infer / Finetune / Train | [BFM_Fitting/](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/BFM_Fitting/) |


**3. Face detection and alignment models**

| Model | Description | Required-by | MindSpore Checkpoint |
| --- | --- | --- | --- |
| gfpgan_weights/ | Face detection and enhanced models used in `facexlib` and `gfpgan`. | Infer / Finetune / Train | [gfpgan_weights/](https://download-mindspore.osinfra.cn/toolkits/mindone/sadtalker/gfpgan_weights/) |


Please download the required weights accordingly (infer/finetune/train). For example, to perform inference, download all weights tagged "Infer". After downloading, the folder should look like this (case-sensitive):

  ```bash
  checkpoints/
  ├── BFM_Fitting
  │   ├── 01_MorphableModel.mat
  │   ├── BFM09_model_info.mat
  │   ├── BFM_exp_idx.mat
  │   ├── BFM_front_idx.mat
  │   ├── Exp_Pca.bin
  │   ├── facemodel_info.mat
  │   ├── select_vertex_id.mat
  │   ├── similarity_Lm3D_all.mat
  │   └── std_exp.txt
  ├── ms
  │   ├── ms_audio2exp.ckpt
  │   ├── ms_audio2pose.ckpt
  │   ├── ms_generator.ckpt
  │   ├── ms_he_estimator.ckpt
  │   ├── ms_kp_extractor.ckpt
  │   ├── ms_mapping.ckpt
  │   ├── ms_mapping_full.ckpt
  │   ├── ms_net_recon.ckpt
  │   ├── ms_vgg19.ckpt # only required for training
  │   ├── ms_wav2lip.ckpt # only required for training
  │   └── ms_hopenet_robust_alpha1.ckpt # only required for training
  gfpgan/
  └── weights
      ├── alignment_WFLW_4HG.ckpt
      ├── detection_Resnet50_Final.ckpt
      ├── GFPGANv1.4.ckpt
      └── parsing_parsenet.ckpt
  ```

 > _NOTE:_  The checkpoint filename or filepath is pre-defined in [sadtalker-infer.yaml](examples\sadtalker\config\sadtalker-infer.yaml). If you want to customize the file name or folder structure, modify the YAML file at the same time.


### Training Data

We use [VoxCeleb](https://mm.kaist.ac.kr/datasets/voxceleb/) data to train SadTalker. Training codes is still under developement. We will release it when it's ready, thanks!


### Example input data for inference

In the original github, there're some example audios and images under [SadTalker/examples](https://github.com/OpenTalker/SadTalker/tree/main/examples). You can download them to quickly start playing Sadtalker! :wink:


## Inference

To generate a talker head video, you have to specify a single portrait image using the argument `--source_image` and an audio file via `--driven_audio`. If you don't specify, it will use the default values.

As reference, you can run the following commands to execute inference process. There're also some arguments to customize the animation, please refer to [input arguments](./utils/arg_parser.py).

```bash
python inference.py --config ./config/sadtalker-infer.yaml --source_image examples/source_image/people_0.png --driven_audio examples/driven_audio/imagine.wav
```

Here are some generated videos with different inputs:

| Chinese audio + full character image   | English audio + full character image       |   Singing audio + character image with cropping preprocessing |
|:--------------------: |:--------------------: | :----: |
| <video  src="https://github.com/hqkate/sadtalker/assets/26082447/fc20924f-9d42-4432-8f7a-2f8094c23662" title="" type="video/mp4"> </video> | <video  src="https://github.com/hqkate/sadtalker/assets/26082447/a2ecbf7d-cde4-4fb7-b6d4-6301b679e75b" type="video/mp4"> </video>  | <video src="https://github.com/hqkate/sadtalker/assets/26082447/2c713067-f64e-45a7-9ce2-bc57f340bdad" type="video/mp4"> </video> |
