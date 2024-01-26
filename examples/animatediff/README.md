# AnimateDiff based on MindSpore

This repository is the MindSpore implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725).

## Features

- [x] Text-to-video generation with AnimdateDiff v2, supporting 16 frames @512x512 resolution on Ascend 910B, 16 frames @256x256 resolution on GPU 3090
- [x] MotionLoRA inference
- [ ] Motion Module Training
- [ ] Motion LoRA Training
- [ ] AnimateDiff v3

## Requirements

```
pip install -r requirements.txt
```

In case `decord` package is not available in your environment, try `pip install eva-decord`.
Instruction on ffmpeg and decord install on EulerOS:
```
1. install ffmpeg 4, referring to https://ffmpeg.org/releases
    wget wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2 --no-check-certificate
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

First, download the torch pretrained weights referring to [torch animatediff checkpoints](https://github.com/guoyww/AnimateDiff/blob/main/__assets__/docs/animatediff.md#download-base-t2i--motion-module-checkpoints).

- Convert SD dreambooth model
```
cd ../examples/stable_diffusion_v2
python tools/model_conversion/convert_weights.py  --source ../animatediff/models/torch_ckpts/toonyou_beta3.safetensors   --target models/toonyou_beta3.ckpt  --model sdv1  --source_version pt
```

- Convert Motion Module
```
cd ../examples/animatediff/tools
python motion_module_convert.py --src ../torch_ckpts/mm_sd_v15_v2.ckpt --tar ../models/motion_module
```

- Convert Motion LoRA
```
cd ../examples/animatediff/tools
python motion_lora_convert.py --src ../torch_ckpts/.ckpt --tar ../models/motion_lora
```


## Inference

### Text-to-Video

- Running On Ascend 910\*:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou.yaml --L 16 --H 512 --W 512
```

By default, DDIM sampling is used, and the sampling speed is 1.07s/iter.

Results:

<p float="left">
<img src=https://github.com/SamitHuang/mindone/assets/8156835/9d6ef65f-223d-407c-bc85-a852d3594934 width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/40dbe614-ccc6-4567-ab53-099cb8d61ebc width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/fb9e2069-041a-4e81-b88e-ccdcfa8afd32 width="25%" />
</p>


- Running on GPU:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou.yaml --L 16 --H 256 --W 256 --device_target GPU
```

### Motion LoRA
- Running On Ascend 910\*:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou-MotionLoRA.yaml --L 16 --H 512 --W 512
```

By default, DDIM sampling is used, and the sampling speed is 1.07s/iter.

Results using Zoom-In motion lora:

<p float="left">
<img src=https://github.com/SamitHuang/mindone/assets/8156835/9357b2e4-0479-4afa-a28b-7a121aba865e width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/f8ff1d2a-20d8-447d-89b2-fd94430db7a4 width="25%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/d4d947a3-4d10-4c7e-b134-a725269037c3 width="25%" />
</p>


- Running on GPU:
```
python text_to_video.py --config configs/prompts/v2/1-ToonYou-MotionLoRA.yaml --L 16 --H 256 --W 256 --device_target GPU
```

## Training

### Image Finetuning

```
python train.py --config configs/training/image_finetune.yaml
```
> For 910B, please set `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` before running training.


### Motion Module Training

```
python train.py --config configs/training/mmv2_train.yaml
```
> For 910B, please set `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` before running training.

You may change the arguments including data path, output directory, lr, etc in the yaml config file. You can also change by command line arguments referring to `args_train.py` or `python train.py --help`

- Evaluation
Inference with the trained model:
```
python text_to_video.py --config configs/prompt/v2/base_video.yaml \
    --motion_module_path {path to saved checkpoint} \
    --prompt  {text prompt}  \
```

You can also create a new config yaml to specify the prompts to test and the motion moduel path based on `configs/prompt/v2/base_video.yaml`.


### Motion LoRA Training

```
python train.py --config configs/training/mmv2_lora.yaml
```
> For 910B, please set `export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"` before running training.


- Inference with the trained model:
```
python text_to_video.py --config configs/prompt/v2/base_video.yaml \
    --motion_lora_path {path to saved checkpoint} \
    --prompt  {text prompt}  \
```


### Training on GPU

Please add `--device_target GPU` in the above training commands and adjust `image_size`/`num_frames`/`train_batch_size` to fit your device memory, if you wan to run on GPUs. Below is an example for 3090.

```
# reduce num frames and batch size to avoid OOM in 3090
python train.py --config configs/training/mmv2_train.yaml --data_path ../videocomposer/datasets/webvid5 --image_size 256 --num_frames=4 --device_target GPU --train_batch_size=1
```
