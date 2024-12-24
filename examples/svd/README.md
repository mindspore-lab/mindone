# Stable Video Diffusion (VideoLDM)

<p align="center"><img width="600" alt="VideoLDM U-Net Block Architecture"
src="https://github.com/mindspore-lab/mindone/assets/16683750/0ac4c83e-f91d-4024-a8e5-e4fdad3f251a"/></p>

## Introduction

Stable Video Diffusion is an Image-to-Video generation model based on Stable Diffusion that extends it to a video
generation task by introducing temporal layers into the architecture (a.k.a. VideoLDM). Additionally, it utilizes a
modified Decoder with added temporal layers to counteract flickering artifacts.

<p align="center"><img width="600" alt="VideoLDM U-Net Block Architecture"
src="https://github.com/mindspore-lab/mindone/assets/16683750/e291f64d-fb49-4983-b488-22d96addb9fb"/>
<br><em>An example of a single U-Net Block with Added Temporal Layers (for more information please refer to <a href="#acknowledgements">[2]</a>)</em></p>

## Requirements

| mindspore | ascend driver | firmware    | cann toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1       |

## Pretrained Models


| SD Base Version | SVD version | Trained for          | Config                      | Checkpoint                                                                                |
|:-----------------:|:-------------:|:----------------------:|:-----------------------------:|:-------------------------------------------------------------------------------------------:|
| v2.0 & v2.1     | SVD         | 14 frames generation | [yaml](configs/svd.yaml)    | [Download (9GB)](https://download.mindspore.cn/toolkits/mindone/svd/svd-d19a808f.ckpt)    |
|                 | SVD-XT      | 25 frames generation | [yaml](configs/svd_xt.yaml) | [Download (9GB)](https://download.mindspore.cn/toolkits/mindone/svd/svd_xt-993f895f.ckpt) |


The weights above were converted from the PyTorch version. If you want to convert another custom model, you can do so by
using `svd_tools/convert.py`. For example:

```shell
python svd_tools/convert.py \
--pt_weights_file PATH_TO_YOUR_TORCH_MODEL \
--config CONFIG_FILE \
--out_dir PATH_TO_OUTPUT_DIR
```

## Inference

Currently, only Image-to-Video generation is supported. For video generation from text, an image must first be created
using either [SD](../stable_diffusion_v2/README.md#inference) or
[SDXL](../stable_diffusion_xl/GETTING_STARTED.md#inference) (recommended resolution is 1024x576).
Once the image is created, the video can be generated using the following command:

```shell
python image_to_video.py --mode=1 \
--SVD.config=configs/svd.yaml \
--SVD.checkpoint=PATH_TO_YOUR_SVD_CHECKPOINT \
--SVD.num_frames=NUM_FRAMES_TO_GENERATE \
--SVD.fps=FPS \
--image=PATH_TO_INPUT_IMAGE
```

> [!TIP]
> If you encounter an OOM error while running the above command, try setting the `--SVD.decode_chunk_size` argument to
> a lower value (default is `num_frames`) before reducing `num_frames` as decoding is very memory-intensive.

For more information on possible parameters and usage, please execute the following command:

```shell
python image_to_video.py --help
```

## Training

### Dataset Preparation

Video labels should be stored in a CSV file in the following format:

```text
path,length,motion_bucket_id
path_to_video1,video_length1,motion_bucket_id1
path_to_video2,video_length2,motion_bucket_id2
...
```

The generation of motion bucket IDs is described in detail in the SVD [[1](#acknowledgements)] paper.
Please refer to Appendix C of the paper for more information.

### Training

Currently, only Image-to-Video generation training is supported.
To train Stable Video Diffusion, execute the following command:

```shell
python train.py --config=configs/svd_train.yaml \
--svd_config=configs/svd.yaml \
--train.pretrained=PATH_TO_YOUR_SVD_CHECKPOINT \
--train.output_dir=PATH_TO_OUTPUT_DIR \
--environment.mode=0 \
--train.temporal_only=True \
--train.epochs=NUM_EPOCHS \
--train.dataset.init_args.frames=NUM_FRAMES \
--train.dataset.init_args.step=FRAMES_FETCHING_STEP \
--train.dataset.init_args.data_dir=PATH_TO_DATASET \
--train.dataset.init_args.metadata=PATH_TO_LABELS
```

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

|  model_name  | cards | batch size  | resolution | grad accu  |sink | jit_level | graph compile | s/step | recipe |
|:-------------:|:-------:|:-------------:|:-------------:|:-----------:|:-----:|:---------:|:--------------:|:---------:|:------:|
|     svd     |    1  |      1     |   4x576x1024 |     5    |  OFF | O0    |     6 mins   |   1.18   | [yaml](configs/svd_train.yaml) |


> [!NOTE]
> More details on the training arguments can be found in the [training config](configs/svd_train.yaml)
> and [model config](configs/svd.yaml).



## Acknowledgements

1. Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion
   English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach. Stable Video Diffusion: Scaling Latent Video
   Diffusion Models to Large Datasets. Stability AI, 2023.
2. Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, Karsten Kreis. Align your
   Latents: High-Resolution Video Synthesis with Latent Diffusion Models. arXiv:2304.08818, 2023.
