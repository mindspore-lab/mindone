# AnimateDiff based on MindSpore

This repository is the MindSpore implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725).

## Features

- [x] Text-to-video generation with AnimdateDiff v2, supporting 16 frames @512x512 resolution on Ascend 910*
- [x] MotionLoRA inference
- [x] Motion Module Training
- [X] Motion LoRA Training
- [X] AnimateDiff v3 Inference

## Requirements

| mindspore  | ascend driver  |  firmware   |cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   2.3.1    |    24.1.RC2    | 7.3.0.1.231 |   8.0.RC2.beta1    |
|   2.2.10   |     23.0.3     | 7.1.0.5.220 |    7.0.0.beta1     |

To install other dependent packages:
```bash
pip install -r requirements.txt
```

In case `decord` package is not available, try `pip install eva-decord`.
For EulerOS, instructions on ffmpeg and decord installation are as follows.

```
1. install ffmpeg 4, referring to https://ffmpeg.org/releases
    wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2 --no-check-certificate
    tar -xvf ffmpeg-4.0.1.tar.bz2
    mv ffmpeg-4.0.1 ffmpeg
    cd ffmpeg
    ./configure --enable-shared         # --enable-shared is needed for sharing libavcodec with decord
    make -j 64
    make install
2. install decord, referring to https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source
    git clone --recursive https://github.com/dmlc/decord
    cd decord
    rm build && mkdir build && cd build
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
    make -j 64
    make install
    cd ../python
    python3 setup.py install --user
```

## Prepare Model Weights

Please download the following weights from to [huggingface](https://huggingface.co/guoyww/animatediff/tree/main/).

The full tree of expected checkpoints is shown below:
```
models
├── domain_adapter_lora
│   └── v3_sd15_adapter.ckpt
├── dreambooth_lora
│   ├── realisticVisionV51_v51VAE.ckpt
│   └── toonyou_beta3.ckpt
├── motion_lora
│   └── v2_lora_ZoomIn.ckpt
├── motion_module
│   ├── mm_sd_v15.ckpt
│   ├── mm_sd_v15_v2.ckpt
│   └── v3_sd15_mm.ckpt
├── sparsectrl_encoder
│   ├── v3_sd15_sparsectrl_rgb.ckpt
│   └── v3_sd15_sparsectrl_scribble.ckpt
└── stable_diffusion
    └── sd_v1.5-d0ab7146.ckpt
```

Then, put all the weights under `animatediff/models/torch_ckpts/` and convert them by running the following command.

```shell
sh scripts/convert_weights.sh
```

## Inference

### Inference (AnimateDiff v3 and SparseCtrl)

```
# download demo images
bash scripts/download_demo_images.sh

# under general T2V setting
python text_to_video.py --config configs/prompts/v3/v3-1-T2V.yaml

# image animation (on RealisticVision)
python text_to_video.py --config configs/prompts/v3/v3-2-animation-RealisticVision.yaml

# sketch-to-animation and storyboarding (on RealisticVision)
python text_to_video.py --config configs/prompts/v3/v3-3-sketch-RealisticVision.yaml
```


Results:

<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Input (by RealisticVision)</td>
    <td width=25% style="border: none; text-align: center">Animation</td>
    <td width=25% style="border: none; text-align: center">Input</td>
    <td width=25% style="border: none; text-align: center">Animation</td>
    </tr>
    <tr>
    <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/__assets__/demos/image/RealisticVision_firework.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/v3/0-closeup-face-photo-of-man-in-black-clothes%2C-night-city.gif" style="width:100%"></td>
    <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/__assets__/demos/image/RealisticVision_sunset.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/v3/0-masterpiece%2C-bestquality%2C-highlydetailed%2C-ultradetailed%2C-sunset%2C-orange-sky%2C-warm-lighting%2C-fishing.gif" style="width:100%"></td>
    </tr>
</table>

<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Input Scribble</td>
    <td width=25% style="border: none; text-align: center">Output</td>
    <td width=25% style="border: none; text-align: center">Input Scribbles</td>
    <td width=25% style="border: none; text-align: center">Output</td>
    </tr>
    <tr>
      <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/__assets__/demos/scribble/scribble_1.png" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/v3/0-a-back-view-of-a-boy%2C-standing-on-the-ground%2C.gif" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/__assets__/demos/scribble/scribble_2_readme.png" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/animatediff/v3/0-an-aerial-view-of-a-modern-city%2C-sunlight%2C-day-time%2C.gif" style="width:100%"></td>
    </tr>
</table>

### Inference (AnimateDiff v2)

#### Text-to-Video

The script uses DDIM sampling by default:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou.yaml --L 16 --H 512 --W 512
```

Results:

<p float="left">
<img src=https://github.com/SamitHuang/mindone/assets/8156835/9d6ef65f-223d-407c-bc85-a852d3594934 width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/40dbe614-ccc6-4567-ab53-099cb8d61ebc width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/fb9e2069-041a-4e81-b88e-ccdcfa8afd32 width="25%" />
</p>

#### Motion LoRA

The script uses DDIM sampling by default:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou-MotionLoRA.yaml --L 16 --H 512 --W 512
```

Results using Zoom-In motion lora:

<p float="left">
<img src=https://github.com/SamitHuang/mindone/assets/8156835/9357b2e4-0479-4afa-a28b-7a121aba865e width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/f8ff1d2a-20d8-447d-89b2-fd94430db7a4 width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/d4d947a3-4d10-4c7e-b134-a725269037c3 width="25%" />
</p>

## Training (AnimateDiff v2)

### Image Finetuning

```
python train.py --config configs/training/image_finetune.yaml
```
> Please set `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` before running train script if using mindspore 2.2.10.

Infer with the trained model by running:

```
python text_to_video.py --config configs/prompts/v2/base_video.yaml \
    --pretrained_model_path {path to saved checkpoint} \
    --prompt  {text prompt}  \
```

### Motion Module Training

```
python train.py --config configs/training/mmv2_train.yaml
```
> Please set `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` before running train script if using mindspore 2.2.10.

You may change the arguments including data path, output directory, lr, etc in the yaml config file. You can also change by command line arguments referring to `args_train.py` or `python train.py --help`

Min-SNR weighting can improve diffusion training convergence. Enable it by appending `--snr_gamma=5.0` to the training command.

Infer with the trained model by running:

```
python text_to_video.py --config configs/prompts/v2/base_video.yaml \
    --motion_module_path {path to saved checkpoint} \
    --prompt  {text prompt}  \
```

You can also create a new config yaml to specify the prompts to test and the motion moduel path based on `configs/prompt/v2/base_video.yaml`.


Here are some generation results after MM training on 512x512 resolution and 16-frame data.

<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Disco light leaks disco ball light reflections shaped rectangular and line with motion blur effect</td>
    <td width=25% style="border: none; text-align: center">Cloudy moscow kremlin time lapse</td>
    <td width=25% style="border: none; text-align: center">Sharp knife to cut delicious smoked fish</td>
    <td width=25% style="border: none; text-align: center">A baker turns freshly baked loaves of sourdough bread</td>
    </tr>
    <tr>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/22fe1fcf-9dbd-4db4-8082-bcec5ce4cc7a" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/01856c0c-cfa9-4445-9c3d-7abc1af245e6" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/eb53baa6-1fb7-44f5-aced-bd7609fca9a2" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/135b552a-7331-478d-9590-f201b1145dff" style="width:100%"></td>
    </tr>
</table>


### Motion LoRA Training

```
python train.py --config configs/training/mmv2_lora.yaml
```

> Please set `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` before running train script if using mindspore 2.2.10.

Infer with the trained model by running:

```
python text_to_video.py --config configs/prompts/v2/base_video.yaml \
    --motion_lora_path {path to saved checkpoint} \
    --prompt  {text prompt}  \
```

Here are some generation results after lora fine-tuning on 512x512 resolution and 16-frame data.

<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Disco light leaks disco ball light reflections shaped rectangular and line with motion blur effect</td>
    <td width=25% style="border: none; text-align: center">Cloudy moscow kremlin time lapse</td>
    <td width=25% style="border: none; text-align: center">Sharp knife to cut delicious smoked fish</td>
    <td width=25% style="border: none; text-align: center">A baker turns freshly baked loaves of sourdough bread</td>
    </tr>
    <tr>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/03d4d494-9ee4-473a-82c4-2d95fecf28f6" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/72075086-6f14-43ec-9a1b-3f27adc3ad4f" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/a4a5ee37-81df-4498-972b-ab454de77fc4" style="width:100%"></td>
      <td width=25% style="border: none"><img src="https://github.com/SamitHuang/mindone/assets/8156835/93b3ba6a-350d-4d35-8e44-d0445d8f3089" style="width:100%"></td>
    </tr>
</table>


## Performance (AnimateDiff v2)

Experiments are tested on ascend 910* graph mode.

### Inference

- mindspore 2.3.1

|   model name   |cards| resolution  | scheduler | steps |     s/step     |    s/video    |
|:--------------:|:--:|:----------: |:---------:|:-----:|:--------------:|:-------------:|
| AnimateDiff v2 |1|  512x512x16 |   DDIM    |  30   |      0.60      |      18.00    |

- mindspore 2.2.10

|   model name   | cards | resolution  | scheduler | steps |   s/step      |    s/video    |
|:--------------:| :--: |:---------:|:-----:|:----------: |:--------------:|:-------------:|
| AnimateDiff v2 | 1 |  512x512x16 |   DDIM    |  30   |      1.20      |      25.00     |

### Training

- mindspore 2.3.1

|             method             |cards|  batch size| resolution    | flash attn | jit level |    s/step    |    img/s    |
|:----------------------------:|:---:|:----------:|:----------:  |:---------------:|:---------:|:------------:|:-------------:|
|         MM training          |  1   |     1     |    16x512x512 |       ON        |    O0     |    1.320     |     0.75      |
|         Motion Lora          |  1   |     1     |    16x512x512 |       ON        |    O0     |    1.566     |     0.64     |
| MM training w/ Embed. cached |  1   |     1     |    16x512x512 |       ON        |    O0     |    1.004     |     0.99     |
| Motion Lora w/ Embed. cached |  1   |     1     |    16x512x512 |       ON        |    O0     |    1.009     |     0.99     |

- mindspore 2.2.10

|             method             |cards| batch size | resolution       | flash attn | jit level | s/step | img/s |
|:----------------------------:|:---:|:----------:|:----------:      |:---------------:|:---------:|:------------:|:-------------:|
|         MM training          |  1   |     1     |    16x512x512     |       OFF       |    N/A     |     1.29     |     0.78     |
|         Motion Lora          |  1   |     1     |    16x512x512     |       OFF       |    N/A     |     1.26     |     0.79     |
| MM training w/ Embed. cached |  1   |     1     |    16x512x512     |       ON        |    N/A     |     0.75     |     1.33     |
| Motion Lora w/ Embed. cached |  1   |     1     |    16x512x512     |       ON        |    N/A     |     0.71     |     1.49     |

> MM training: Motion Module training.

> Embed. cached: The video embedding (VAE-encoder outputs) and text embedding are pre-computed and stored before diffusion training.
