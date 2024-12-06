

## Open-Sora: Democratizing Efficient Video Production for All

Here we provide an efficient MindSpore implementation of [OpenSora](https://github.com/hpcaitech/Open-Sora), an open-source project that aims to foster innovation, creativity, and inclusivity within the field of content creation.

This repository is built on the models and code released by HPC-AI Tech. We are grateful for their exceptional work and generous contribution to open source.

<h4>Open-Sora is still at an early stage and under active development.</h4>



## 📰 News & States

| Official News from HPC-AI Tech                                                                                                                                                                                                                                                                                                                                                                                                                              | MindSpore Support                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **[2024.06.17]** 🔥 HPC-AI released **Open-Sora 1.2**, which includes **3D-VAE**, **rectified flow**, and **score condition**. The video quality is greatly improved. [[checkpoints]](#model-weights) [[report]](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md) [[blog]](https://hpc-ai.com/blog/open-sora-from-hpc-ai-tech-team-continues-open-source-generate-any-16-second-720p-hd-video-with-one-click-model-weights-ready-to-use) | Text-to-Video                                                                                  |
| **[2024.04.25]** 🤗 HPC-AI Tech released the [Gradio demo for Open-Sora](https://huggingface.co/spaces/hpcai-tech/open-sora) on Hugging Face Spaces.                                                                                                                                                                                                                                                                                                        | N.A.                                                                                           |
| **[2024.04.25]** 🔥 HPC-AI Tech released **Open-Sora 1.1**, which supports **2s~15s, 144p to 720p, any aspect ratio** text-to-image, **text-to-video, image-to-video, video-to-video, infinite time** generation. In addition, a full video processing pipeline is released. [[checkpoints]]() [[report]](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_02.md)                                                                               | Image/Video-to-Video; Infinite time generation; Variable resolutions, aspect ratios, durations |
| **[2024.03.18]** HPC-AI Tech released **Open-Sora 1.0**, a fully open-source project for video generation.                                                                                                                                                                                                                                                                                                                                                  | ✅ VAE + STDiT training and inference                                                           |
| **[2024.03.04]** HPC-AI Tech Open-Sora provides training with 46% cost reduction [[blog]](https://hpc-ai.com/blog/open-sora)                                                                                                                                                                                                                                                                                                                                | ✅ Parallel training on Ascend devices                                                          |


## Requirements

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:     |   :---:       | :---:    | :---:              |
| 2.3.1     |  23.0.3     |7.1.0.9.220    |   8.0.RC2.beta1   |



## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910*.

### OpenSora 1.2 Demo

| 4s 720×1280                                                                                     | 4s 720×1280                                                                                     | 4s 720×1280                                                                                     |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| <video src="https://github.com/user-attachments/assets/7d9c812b-1642-4019-99da-dabf94c41596" /> | <video src="https://github.com/user-attachments/assets/9f463262-9ee0-4931-9d39-63fe925cbe6e" /> | <video src="https://github.com/user-attachments/assets/e0fa61bd-8bd0-40aa-9ea6-c587d492482a" /> |

> [!TIP]
> To generate better looking videos, you can try generating in two stages: Text-to-Image and then Image-to-Video.

### OpenSora 1.1 Demo

<details>
<summary>Demo</summary>

#### Text-to-Video

| 16x256x720                                                                                                                                                                                                                                                                                                                                                                      | 16x640x360                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="500" src="https://github.com/mindspore-lab/mindone/assets/16683750/8a78ead9-6d55-4c16-a2c5-7379ddb6542f"/>                                                                                                                                                                                                                                                          | <img height="300" src="https://github.com/mindspore-lab/mindone/assets/16683750/b9e2568c-e238-480e-8a32-5de189c3c1da"/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Snow falling over multiple houses and trees on winter landscape against night sky. christmas festivity and celebration concept                                                                                                                                                                                                                                                  | Snow falling over multiple houses and trees on winter landscape against night sky. christmas festivity and celebration concept                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| <img width="500" src="https://github.com/mindspore-lab/mindone/assets/16683750/c6a4e3f2-751c-42d2-a542-102c362bcc57"/>                                                                                                                                                                                                                                                          | <img height="300" src="https://github.com/mindspore-lab/mindone/assets/16683750/21b68e61-1f19-47dc-a055-25e20e6257a9"/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Time-Lapse Milky Way above the Mountain                                                                                                                                                                                                                                                                                                                                         | Time-Lapse Milky Way above the Mountain                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| <img width="500" src="https://github.com/mindspore-lab/mindone/assets/16683750/09c59cdc-a497-4db9-b55b-f5ea427346e1"/>                                                                                                                                                                                                                                                          | <img height="300" src="https://github.com/mindspore-lab/mindone/assets/16683750/e34e98be-b4f1-40b6-96c6-8722c9454b72"/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Time Lapse of the rising sun over a tree in an open rural landscape, with clouds in the blue sky beautifully playing with the rays of light                                                                                                                                                                                                                                     | A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect. |
| <img width="500" src="https://github.com/mindspore-lab/mindone/assets/16683750/5fa0643b-3ff2-44ef-8463-3cad75a4cb4d"/>                                                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| This close-up shot of a Victoria crowned pigeon showcases its striking blue plumage and red chest. Its crest is made of delicate, lacy feathers, while its eye is a striking red color. The bird’s head is tilted slightly to the side, giving the impression of it looking regal and majestic. The background is blurred, drawing attention to the bird’s striking appearance. |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |

#### Image-to-Video

| Input                                                                                                                                                                                                            | Output                                                                                                                 |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/16683750/069a1b73-0849-4080-8389-906968da68b6"/><br/>a brown bear in the water with a fish in its mouth</p>              | <img width="300" src="https://github.com/mindspore-lab/mindone/assets/16683750/e07ea85f-ec1d-4371-a91c-050b577beb57"/> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/16683750/d85f0ea0-fe21-4001-a8a6-2711f1903da1"/><br/>a group of statues on the side of a building, camera pans right</p> | <img width="300" src="https://github.com/mindspore-lab/mindone/assets/16683750/b487adb8-8625-48a0-b8f8-6584463639ed"/> |

#### Frame Interpolation

| Start Frame                                                                                                            | End Frame                                                                                                              | Caption                       | Output                                                                                                                 |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------|
| <img width="400" src="https://github.com/mindspore-lab/mindone/assets/16683750/77416ccc-ab7e-4c1e-be2a-e50777f2e0f1"/> | <img width="400" src="https://github.com/mindspore-lab/mindone/assets/16683750/1f32eb12-b2e7-4066-89f0-48516cbd9581"/> | A breathtaking sunrise scene. | <img width="400" src="https://github.com/mindspore-lab/mindone/assets/16683750/faf10f9b-4472-4fb7-a505-440deb40b14b"/> |

#### Video Editing

| Input                                                                                                                                                           | Output                                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| <p align="center"><img width="400" src="https://github.com/mindspore-lab/mindone/assets/16683750/0863676c-345b-4b91-82f4-c8c835e2b562"/><br/>a snowy forest</p> | <img width="400" src="https://github.com/mindspore-lab/mindone/assets/16683750/9086e7a6-5844-4804-877d-c0df481999c4"/> |

#### Text-to-Image

| Caption                                                                                                           | Output                                                                                                                                       |
|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Bright scene, aerial view,ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens. | <img width="400" src="https://github.com/mindspore-lab/mindone/assets/16683750/a3ad7ada-6a2e-4071-9503-c4851547973e"/>                       |
| A small cactus with a happy face in the Sahara desert.                                                            | <p align="center"><img width="250" src="https://github.com/mindspore-lab/mindone/assets/16683750/46a1e8b1-abb5-46b7-9fb5-813c35e1b29e"/></p> |

</details>

### OpenSora 1.0 Demo

<details>
<summary>Demo</summary>

| **2s 512×512**                                                                                                                                                                 | **2s 512×512**                                                                                                                                                              | **2s 512×512**                                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ![009-A-serene-night-scene-in-a-forested-area -The-first](https://github.com/SamitHuang/mindone/assets/8156835/72f0dd45-bcf5-47b2-b2b3-24599bd9b16e)                           | ![000-A-soaring-drone-footage-captures-the-majestic-beauty-of-a](https://github.com/SamitHuang/mindone/assets/8156835/6bde280b-80a7-4617-a53d-58981ef308c2)                 | ![001-A-majestic-beauty-of-a-waterfall-cascading-down-a-cliff](https://github.com/SamitHuang/mindone/assets/8156835/a0b5d303-71d7-4de0-9592-0784bac398bf)           |
| A serene night scene in a forested area. [...] The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. | A soaring drone footage captures the majestic beauty of a coastal cliff, [...] The water gently laps at the rock base and the greenery that clings to the top of the cliff. | The majestic beauty of a waterfall cascading down a cliff into a serene lake. [...] The camera angle provides a bird's eye view of the waterfall.                   |
| ![006-A-bustling-city-street-at-night,-filled-with-the-glow](https://github.com/SamitHuang/mindone/assets/8156835/00a966c8-16fa-4799-98a6-3d69c2983e49)                        | ![002-A-vibrant-scene-of-a-snowy-mountain-landscape -The-sky](https://github.com/SamitHuang/mindone/assets/8156835/fb243b36-b2dd-4bac-a8b2-812b5c3b35da)                    | ![004-A-serene-underwater-scene-featuring-a-sea-turtle-swimming-through](https://github.com/SamitHuang/mindone/assets/8156835/31a7f201-b436-4a85-a68c-e0cd58d8bca5) |
| A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. [...]                                                           | The vibrant beauty of a sunflower field. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. [...]                                            | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell [...]                                     |

Videos are downsampled to `.gif` for display. Click for original videos. Prompts are trimmed for display, see [here](assets/texts/t2v_samples.txt) for full prompts.

</details>


## 🔆 Features

- 📍 **Open-Sora 1.2** released. Model weights are available [here](#model-weights). See [report 1.2](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md) for more details.
    - ✅ Support rectified flow scheduling.
    - ✅ Support more conditioning including fps, aesthetic score, motion strength and camera motion.
    - ✅ Trained our 3D-VAE for temporal dimension compression.

- 📍 **Open-Sora 1.1** with the following features
    - ✅ Improved ST-DiT architecture includes Rotary Position Embedding (RoPE), QK Normalization, longer text length, etc.
    - ✅ Support image and video conditioning and video editing, and thus support animating images, connecting videos, etc.
    - ✅ Support training with any resolution, aspect ratio, and duration.

- 📍 **Open-Sora 1.0** with the following features
    - ✅ Text-to-video generation in 256x256 or 512x512 resolution and up to 64 frames.
    - ✅ Three-stage training: i) 16x256x256 video pretraining, ii) 16x512x512 video fine-tuning, and iii) 64x512x512 videos
    - ✅ Optimized training recipes for MindSpore+Ascend framework (see `configs/opensora/train/xxx_ms.yaml`)
    - ✅ Acceleration methods: flash attention, recompute (gradient checkpointing), data sink, mixed precision, and graph compilation.
    - ✅ Data parallelism + Optimizer parallelism, allow training on 300x512x512 videos

<details>
<summary>View more</summary>

* ✅ Following the findings in OpenSora, we also adopt the VAE from Stable Diffusion for video latent encoding.
* ✅ We pick the **STDiT** model as our video diffusion transformer following the best practice in OpenSora.
* ✅ Support T5 text conditioning.

</details>

<details>
<summary>View more</summary>

* [ ] Evaluation pipeline.
* [ ] Complete the data processing pipeline (including dense optical flow, aesthetics scores, text-image similarity, etc.).

</details>

## Contents

* [Installation](#installation)
* [Model Weights](#model-weights)
* [Inference](#inference)
* [Data Processing](#data-processing)
* [Training](#training)
* [Evaluation](#evaluation)
* [VAE Training & Evaluation](#vae-training--evaluation)
* [Long sequence training and inference (sequence parallel)](#long-sequence-training-and-inference-sequence-parallel)
* [Contribution](#contribution)
* [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

* Repo structure: [structure.md](docs/structure.md)


## Installation

1. Please install MindSpore 2.3.1 according to the [MindSpore official website](https://www.mindspore.cn/install/) and install [CANN 8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) as recommended by the official installation website.

2. Install requirements
```bash
pip install -r requirements.txt
```

In case `decord` package is not available, try `pip install eva-decord`.
For EulerOS, instructions on ffmpeg and decord installation are as follows.

<details onclose>

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

</details>

## Model Weights

### Open-Sora 1.2 Model Weights

| Model              | Model size | Data | URL                                                             |
|--------------------|------------|------|-----------------------------------------------------------------|
| STDiT3 (Diffusion) | 1.1B       | 30M  | [Download](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3) |
| VAE                | 384M       | 3M   | [Download](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2) |

- Convert STDiT3 to MS checkpoint:

```shell
python tools/convert_pt2ms.py --src /path/to/OpenSora-STDiT-v3/model.safetensors --target models/opensora_stdit_v3.ckpt
```

- Convert VAE to MS checkpoint:

```shell
python convert_vae_3d.py --src /path/to/OpenSora-VAE-v1.2/model.safetensors --target models/OpenSora-VAE-v1.2/model.ckpt
```

- The T5 model is identical to OpenSora 1.0 and can be downloaded from the links below.


### Open-Sora 1.1 Model Weights

<details>
<summary>Instructions</summary>

- STDit:

| Stage | Resolution         | Model Size | Data                       | #iterations | URL                                                                    |
|-------|--------------------|------------|----------------------------|-------------|------------------------------------------------------------------------|
| 2     | mainly 144p & 240p | 700M       | 10M videos + 2M images     | 100k        | [Download](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2) |
| 3     | 144p to 720p       | 700M       | 500K HQ videos + 1M images | 4k          | [Download](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3) |

Convert to MS checkpoint:

```
python tools/convert_pt2ms.py --src /path/to/OpenSora-STDiT-v2-stage3/model.safetensors --target models/opensora_v1.1_stage3.ckpt
```

- T5 and VAE models are identical to OpenSora 1.0 and can be downloaded from the links below.

</details>

### Open-Sora 1.0 Model Weights

<details>
<summary>Instructions</summary>

Please prepare the model checkpoints of T5, VAE, and STDiT and put them under `models/` folder as follows.

- T5: Download the [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main) folder and put it under `models/`

    Convert to ms checkpoint:
    ```
    python tools/convert_t5.py --src models/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin  models/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin --target models/t5-v1_1-xxl/model.ckpt

    ```

- VAE: Download the safetensor checkpoint from [here]((https://huggingface.co/stabilityai/sd-vae-ft-ema/tree/main))

    Convert to ms checkpoint:
    ```
    python tools/convert_vae.py --src /path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors --target models/sd-vae-ft-ema.ckpt
    ```

- STDiT: Download `OpenSora-v1-16x256x256.pth` / `OpenSora-v1-HQ-16x256x256.pth` / `OpenSora-v1-HQ-16x512x512.pth` from [here](https://huggingface.co/hpcai-tech/Open-Sora/tree/main)

    Convert to ms checkpoint:

    ```
    python tools/convert_pt2ms.py --src /path/to/OpenSora-v1-16x256x256.pth --target models/OpenSora-v1-16x256x256.ckpt
    ```

    Training orders: 16x256x256 $\rightarrow$ 16x256x256 HQ $\rightarrow$ 16x512x512 HQ.

    These model weights are partially initialized from [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha). The number of
parameters is 724M. More information about training can be found in HPC-AI Tech's **[report](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md)**. More about the dataset can be found in [datasets.md](https://github.com/hpcaitech/Open-Sora/blob/main/docs/datasets.md) from HPC-AI Tech. HQ means high quality.

- PixArt-α: Download the pth checkpoint from [here](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-512x512.pth) (for training only)

    Convert to ms checkpoint:
    ```
    python tools/convert_pt2ms.py --src /path/to/PixArt-XL-2-512x512.pth --target models/PixArt-XL-2-512x512.ckpt
    ```

</details>



## Inference

### Open-Sora 1.2 and 1.1 Command Line Inference

#### Image/Video-to-Video Generation (supports text guidance)

```shell
# OSv1.2
python scripts/inference.py --config configs/opensora-v1-2/inference/sample_iv2v.yaml --ckpt_path /path/to/your/opensora-v1-1.ckpt
# OSv1.1
python scripts/inference.py --config configs/opensora-v1-1/inference/sample_iv2v.yaml --ckpt_path /path/to/your/opensora-v1-1.ckpt
```
> For parallel inference, please use `mpirun` or `msrun`, and append `--use_parallel=True` to the inference script referring to `scripts/run/run_infer_os_v1.1_t2v_parallel.sh`

In the `sample_iv2v.yaml`, provide such information as `loop`, `condition_frame_length`, `captions`, `mask_strategy`,
and `reference_path`.
See [here](docs/quick_start.md#imagevideo-to-video-opensora-v11-and-above) for more details.

> For inference with sequence parallelism using multiple NPUs in Open-Sora 1.2, please use `msrun` and append `--use_parallel True` and `--enable_sequence_parallelism True` to the inference script, referring to `scripts/run/run_infer_sequence_parallel.sh`. To further accelerate the inference speed, you can use [DSP](https://arxiv.org/abs/2403.10266) by appending `--dsp True`, referring to `scripts/run/run_infer_sequence_parallel_dsp.sh`.

#### Text-to-Video Generation

To generate a video from text, you can use `sample_t2v.yaml` or set `--reference_path` to an empty string `''`
when using `sample_iv2v.yaml`.

```shell
python scripts/inference.py --config configs/opensora-v1-1/inference/sample_t2v.yaml --ckpt_path /path/to/your/opensora-v1-1.ckpt
```

#### Inference Performance

We evaluate the inference performance of text-to-video generation by measuring the average sampling time per step and the total sampling time of a video.

All experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.


| model name      |  cards | batch size | resolution |  jit level | precision |  scheduler   | step      | graph compile | s/step     | s/video | recipe |
| :--:         | :--:   | :--:       | :--:       | :--:       | :--:       | :--:       | :--:       | :--:      |:--:    | :--:   |:--:   |
| STDiT2-XL/2  |  1     | 1          | 16x640x360   | O0       | bf16       |  DDPM     |   100   |  1~2 mins |  1.56    |    156.00      |  [yaml](configs/opensora-v1-1/inference/sample_t2v.yaml) |
| STDiT3-XL/2  |  1     | 1          | 51x720x1280   | O0      | bf16       |  RFlow    |   30    |  1~2 mins  |  5.88      |  176.40   | [yaml](configs/opensora-v1-2/inference/sample_t2v.yaml) |
| STDiT3-XL/2  |  1     | 1          | 102x720x1280  | O0      | bf16       |  RFlow    |   30    |  1~2 min   | 13.71      |  411.30  | [yaml](configs/opensora-v1-2/inference/sample_t2v.yaml) |



### Open-Sora 1.0 Command Line Inference

<details>
<summary>Instructions</summary>

You can run text-to-video inference via the script `scripts/inference.py` as follows.

```bash
# Sample 16x256x256 videos
python scripts/inference.py --config configs/opensora/inference/stdit_256x256x16.yaml --ckpt_path models/OpenSora-v1-HQ-16x256x256.ckpt --prompt_path /path/to/prompt.txt

# Sample 16x512x512 videos
python scripts/inference.py --config configs/opensora/inference/stdit_512x512x16.yaml --ckpt_path models/OpenSora-v1-HQ-16x512x512.ckpt --prompt_path /path/to/prompt.txt

# Sample 64x512x512 videos
python scripts/inference.py --config configs/opensora/inference/stdit_512x512x64.yaml --ckpt_path /path/to/your/opensora-v1.ckpt --prompt_path /path/to/prompt.txt
```
> For parallel inference, please use `mpirun` or `msrun`, and append `--use_parallel=True` to the inference script referring to `scripts/run/run_infer_t2v_parallel.sh`

We also provide a three-stage sampling script `run_sole_3stages.sh` to reduce memory limitation, which decomposes the whole pipeline into text embedding, text-to-video latent sampling, and vae decoding.

For more usage on the inference script, please run `python scripts/inference.py -h`

#### Inference Performance

We evaluate the inference performance of text-to-video generation by measuring the average sampling time per step and the total sampling time of a video.

All experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name      |  cards | batch size | resolution |  jit level | precision |  scheduler   | step      | graph compile | s/step     | s/video | recipe |
| :--:         | :--:   | :--:       | :--:       | :--:       | :--:       | :--:       | :--:       | :--:      |:--:    | :--:   |:--:   |
| STDiT-XL/2 | 1 | 4 | 16x256x256 |  O0 | fp32 | DDPM | 100 |  2~3 mins | 0.39 | 39.22 | [yaml](configs/opensora/inference/stdit_256x256x16.yaml) |
| STDiT-XL/2 | 1 | 1 | 16x512x512 | O0 | fp32 | DDPM | 100 | 2~3 mins |  1.85 | 185.00 | [yaml](configs/opensora/inference/stdit_512x512x16.yaml) |
| STDiT-XL/2 | 1 | 1 | 64x512x512 |  O0 | bf16 | DDPM | 100 | 2~3 mins | 2.78 | 278.45 | [yaml](configs/opensora/inference/stdit_512x512x64.yaml) |

</details>

<br>

> ⚠️ Note: When running parallel inference scripts under `scripts/run/` on ModelArts, please `unset RANK_TABLE_FILE` before the inference starts.

## Data Processing

Currently, we are developing the complete pipeline for data processing from raw videos to high-quality text-video pairs. We provide the data processing tools as follows.

<details>
<summary>View more</summary>

The text-video pair data should be organized as follows, for example.

```text
.
├── video_caption.csv
├── video_folder
│   ├── part01
│   │   ├── vid001.mp4
│   │   ├── vid002.mp4
│   │   └── ...
│   └── part02
│       ├── vid001.mp4
│       ├── vid002.mp4
│       └── ...
```

The `video_folder` contains all the video files. The csv file `video_caption.csv` records the relative video path and its text caption in each line, as follows.

```text
video,caption
video_folder/part01/vid001.mp4,a cartoon character is walking through
video_folder/part01/vid002.mp4,a red and white ball with an angry look on its face
```

### Cache Text Embeddings

For acceleration, we pre-compute the t5 embedding before training stdit.

```bash
python scripts/infer_t5.py \
    --csv_path /path/to/video_caption.csv \
    --output_path /path/to/text_embed_folder \
    --model_max_length 300     # 300 for OpenSora v1.2, 200 for OpenSora v1.1, 120 for OpenSora 1.0
```

OpenSora v1 uses text embedding sequence length of 120 (by default).
If you want to generate text embeddings for OpenSora v1.1, please change `model_max_length` to 200.

After running, the text embeddings saved as npz file for each caption will be in `output_path`. Please change `csv_path` to your video-caption annotation file accordingly.

### Cache Video Embedding (Optional)

If the storage budget is sufficient, you may also cache the video embedding by

```bash
python scripts/infer_vae.py \
    --csv_path /path/to/video_caption.csv  \
    --video_folder /path/to/video_folder  \
    --output_path /path/to/video_embed_folder  \
    --vae_checkpoint models/sd-vae-ft-ema.ckpt \
    --image_size 512 \
```
> for parallel running, please refer to `scripts/run/run_infer_vae_parallel.sh`

For more usage, please check `python scripts/infer_vae.py -h`

After running, the vae latents saved as npz file for each video will be in `output_path`.

Finally, the training data should be like follows.
```text
.
├── video_caption.csv
├── video_folder
│   ├── part01
│   │   ├── vid001.mp4
│   │   ├── vid002.mp4
│   │   └── ...
│   └── part02
│       ├── vid001.mp4
│       ├── vid002.mp4
│       └── ...
├── text_embed_folder
│   ├── part01
│   │   ├── vid001.npz
│   │   ├── vid002.npz
│   │   └── ...
│   └── part02
│       ├── vid001.npz
│       ├── vid002.npz
│       └── ...
├── video_embed_folder  # optional
│   ├── part01
│   │   ├── vid001.npz
│   │   ├── vid002.npz
│   │   └── ...
│   └── part02
│       ├── vid001.npz
│       ├── vid002.npz
│       └── ...

```

Each npz file contains data for the following keys:
- `latent_mean` mean of vae latent distribution
- `latent_std`: std of vae latent distribution
- `fps`: video fps
- `ori_size`: original size (h, w) of the video

After caching VAE, you can use them for STDiT training by parsing `--vae_latent_folder=/path/to/video_embed_folder` to the training script `python train.py`.

#### Cache VAE for multi-resolutions (for OpenSora 1.1)

If there are multiple folders named in `latent_{h}x{w}` format under the `--vae_latent_folder` folder (which is parsed to train.py), one of resolutions will selected randomly during training. For example:

```
video_embed_folder
   ├── latent_576x1024
   │   ├── vid001.npz
   │   ├── vid002.npz
   │   └── ...
   └── latent_1024x576
       ├── vid001.npz
       ├── vid002.npz
       └── ...
```

</details>

## Training

### Open-Sora 1.2

Once you prepare the data in a csv file, you may run the following commands to launch training on a single card.

```shell
# standalone training for stage 2
export MS_DEV_ENABLE_KERNEL_PACKET=on

python scripts/train.py --config configs/opensora-v1-2 /train/train_stage2.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video_folder \
    --text_embed_folder /path/to/text_embed_folder \
```

`text_embed_folder` is required and used to speed up the training. You can find the instructions on how to generate T5 embeddings [here](#cache-text-embeddings).

For parallel training, use `msrun` and along with `--use_parallel=True`:

```shell
# distributed training for stage 2
export MS_DEV_ENABLE_KERNEL_PACKET=on

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    python scripts/train.py --config configs/opensora-v1-2/train/train_stage2.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video_folder \
    --text_embed_folder /path/to/text_embed_folder \
    --use_parallel True
```

You can modify the training configuration, including hyper-parameters and data settings, in the yaml file specified by the `--config` argument.

#### Multi-Resolution Training

OpenSora v1.2 supports training with multiple resolutions, aspect ratios, and frames based on the [bucket method](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_02.md#support-for-multi-timeresolutionaspect-ratiofps-training).

To enable dynamic training for STDiT3, please set the `bucket_config` to fit your datasets and tasks at first. An example (from `configs/opensora-v1-2/train/train_stage2.yaml`) is

```python
bucket_config:
  # Structure: "resolution": { num_frames: [ keep_prob, batch_size ] }
  "144p": { 1: [ 1.0, 475 ], 51: [ 1.0, 51 ], 102: [ [ 1.0, 0.33 ], 27 ], 204: [ [ 1.0, 0.1 ], 13 ], 408: [ [ 1.0, 0.1 ], 6 ] }
  "256": { 1: [ 0.4, 297 ], 51: [ 0.5, 20 ], 102: [ [ 0.5, 0.33 ], 10 ], 204: [ [ 0.5, 1.0 ], 5 ], 408: [ [ 0.5, 1.0 ], 2 ] }
  "240p": { 1: [ 0.3, 297 ], 51: [ 0.4, 20 ], 102: [ [ 0.4, 0.33 ], 10 ], 204: [ [ 0.4, 1.0 ], 5 ], 408: [ [ 0.4, 1.0 ], 2 ] }
  "360p": { 1: [ 0.5, 141 ], 51: [ 0.15, 8 ], 102: [ [ 0.3, 0.5 ], 4 ], 204: [ [ 0.3, 1.0 ], 2 ], 408: [ [ 0.5, 0.5 ], 1 ] }
  "512": { 1: [ 0.4, 141 ], 51: [ 0.15, 8 ], 102: [ [ 0.2, 0.4 ], 4 ], 204: [ [ 0.2, 1.0 ], 2 ], 408: [ [ 0.4, 0.5 ], 1 ] }
  "480p": { 1: [ 0.5, 89 ], 51: [ 0.2, 5 ], 102: [ 0.2, 2 ], 204: [ 0.1, 1 ] }
  "720p": { 1: [ 0.1, 36 ], 51: [ 0.03, 1 ] }
  "1024": { 1: [ 0.1, 36 ], 51: [ 0.02, 1 ] }
  "1080p": { 1: [ 0.01, 5 ] }
  "2048": { 1: [ 0.01, 5 ] }
```

Knowing that the optimal bucket config can varies from device to device, we have tuned and provided bucket config that are more balanced on Ascend + MindSpore in `configs/opensora-v1-2/train/{stage}_ms.yaml`. You may use them for better training performance.

More details on the bucket configuration can be found in [Multi-resolution Training with Buckets](./docs/quick_start.md#4-multi-resolution-training-with-buckets-opensora-v11-and-above).

The instruction for launching the dynamic training task is smilar to the previous section. An example running script is `scripts/run/run_train_os1.2_stage2.sh`.


### Open-Sora 1.1

<details>
<summary>Instructions</summary>

Once you prepare the data in a csv file, you may run the following commands to launch training on a single card.

```shell
# standalone training for stage 1
python scripts/train.py --config configs/opensora-v1-1/train/train_stage1.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video_folder \
    --text_embed_folder /path/to/text_embed_folder \
    --vae_latent_folder /path/to/video_embed_folder
```

`text_embed_folder` and `vae_latent_folder` are optional and used to speed up the training.
You can find more in [T5 text embeddings](#cache-text-embeddings) and [VAE Video Embeddings](#cache-video-embedding-optional)

For parallel training, use `msrun` and along with `--use_parallel=True`:

```shell
# distributed training for stage 1
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    python scripts/train.py --config configs/opensora-v1-1/train/train_stage1.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video_folder \
    --text_embed_folder /path/to/text_embed_folder \
    --vae_latent_folder /path/to/video_embed_folder \
    --use_parallel True
```

#### Multi-Resolution Training

OpenSora v1.1 supports training with multiple resolutions, aspect ratios, and a variable number of frames.
This can be enabled in one of two ways:

1. Provide variable sized VAE embeddings with the `--vae_latent_folder` option.
2. Use `bucket_config` for training with videos in their original format. More on the bucket configuration can be found
   in [Multi-resolution Training with Buckets](./docs/quick_start.md#4-multi-resolution-training-with-buckets-opensora-v11-and-above).

Detailed running command can be referred in `scripts/run/run_train_os_v1.1_stage2.sh`

</details>


### Open-Sora 1.0 Training

<details>
<summary>Instructions</summary>

Once the training data including the [T5 text embeddings](#cache-text-embeddings) is prepared, you can run the following commands to launch training.

```bash
# standalone training, 16x256x256
python scripts/train.py --config configs/opensora/train/stdit_256x256x16_ms.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video_folder \
    --text_embed_folder /path/to/text_embed_folder \
```
> To use the cached video embedding, please replace `--video_folder` with `--video_embed_folder` and pass the path to the video embedding folder.

For parallel training, please use `msrun` and pass `--use_parallel=True`

```bash
# 8 NPUs, 64x512x512
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    python scripts/train.py --config configs/opensora/train/stdit_512x512x64_ms.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video_folder \
    --text_embed_folder /path/to/text_embed_folder \
    --use_parallel True \
```

</details>

To train in bfloat16 precision, please parse `--global_bf16=True`

For more usage, please check `python scripts/train.py -h`.
You may also see the example shell scripts in `scripts/run` for quick reference.


## Evaluation

### Open-Sora 1.2

Open-Sora 1.2 based on MindSpore and Ascend 910* supports 0s\~16s, 144p to 720p, various aspect ratios video generation. The supported configurations are listed below.

|      | image | 2s  | 4s  | 8s  | 16s |
| ---- | ----- | --- | --- | --- | --- |
| 240p | ✅     | ✅   | ✅   | ✅   | ✅   |
| 360p | ✅     | ✅   | ✅   | ✅   | ✅   |
| 480p | ✅     | ✅   | ✅   | ✅   | 🆗   |
| 720p | ✅     | ✅   | ✅   | 🆗   | 🆗   |

Here ✅ means that the data is seen during training, and 🆗 means although not trained, the model can inference at that config. Inference for 🆗 requires sequence parallelism.


#### Training Performance

We evaluate the training performance of Open-Sora v1.2 on the MixKit dataset with high-resolution videos (1080P, duration 12s to 100s).

All experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.
| model name   | cards  | batch size | resolution | precision  | sink      | jit level | graph compile |  s/step | recipe |
| :--:         | :--:   | :--:       | :--:       | :--:       | :--:      | :--:      |:--:          | :--:       | :--:   |
| STDiT3-XL/2  |  8     | 1          | 51x720x1280| bf16       | ON      | O1        |    12 mins   | 14.23   | [yaml](configs/opensora-v1-2/train/train_720x1280x51.yaml)
| STDiT3-XL/2  |  8     | dynamic    | stage 1 | bf16       |   OFF    | O1        |      22 mins   | 13.17   | [yaml](configs/opensora-v1-2/train/train_stage1_ms.yaml)
| STDiT3-XL/2  |  8     | dynamic    | stage 2 | bf16       |   OFF    | O1        |     22 mins     | 31.04   | [yaml](configs/opensora-v1-2/train/train_stage2_ms.yaml)
| STDiT3-XL/2  |  8     | dynamic    | stage 3 | bf16       |   OFF    | O1        |     22 mins     | 31.17   | [yaml](configs/opensora-v1-2/train/train_stage3_ms.yaml)

Note that the step time of dynamic training can be influenced by the resolution and duration distribution of the source videos.

To reproduce the above performance, you may refer to `scripts/run/run_train_os1.2_720x1280x51.sh` and  `scripts/run/run_train_os1.2_stage2.sh`.

Below are some generation results after fine-tuning STDiT3 with **Stage 2** bucket config on a mixkit subset, which contains 100 text-video pairs. The training set contains 80 1080P videos consisting of natural scenes, flowers, and pets. Here we show the text-to-video generation results on the test set.

<table class="center">
<tr>
  <td width=50% style="text-align:center;"><b>480x854x204</b></td>
  <td width=50% style="text-align:center;"><b>480x854x204</b></td>
  </tr>
<tr>
  <td width=50%><video src="https://github.com/user-attachments/assets/e90a82c3-f7d0-43d7-b643-a32de934b9e7" autoplay></td>
  <td width=50%><video src="https://github.com/user-attachments/assets/bbb520db-1a3a-4503-a072-e57389a50ecc" autoplay></td>
</tr>
<tr>
  <td width=50% style="text-align:center;"><b>480x854x204</b></td>
  <td width=50% style="text-align:center;"><b>480x854x204</b></td>
  </tr>
<tr>
  <td width=50%><video src="https://github.com/user-attachments/assets/d79d3cf4-f7c7-4825-ba34-57fea6d1164a" autoplay></td>
  <td width=50%><video src="https://github.com/user-attachments/assets/835604e6-c823-4b59-a214-993cdb873b66" autoplay></td>
</tr>
</table>


### Open-Sora 1.1

<details>
<summary>View more</summary>

#### Training Performance

We evaluate the training performance of Open-Sora v1.1 on a subset of the MixKit dataset.

All experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name   | cards  | batch size | resolution   | vae cache  | precision  | sink       | jit level    | graph compile | s/step    | recipe |
| :--:         | :--:   | :--:       | :--:         | :--:       | :--:       | :--:       | :--:         | :--:          | :--:      | :--:   |
| STDiT3-XL/2  |  8     | 1          | 16x512x512   | OFF        | bf16       | OFF        | O1           |  13 mins      | 2.28      | [yaml](configs/opensora-v1-1/train/stdit2_512x512x64.yaml) |
| STDiT3-XL/2  |  8     | 1          | 64x512x512   | OFF        | bf16       | OFF        | O1           |  13 mins      | 8.57      | [yaml](configs/opensora-v1-1/train/stdit2_512x512x64.yaml) |
| STDiT3-XL/2  |  8     | 1          | 24x576x1024  | OFF        | bf16       | OFF        | O1           |  13 mins      | 8.55      | [yaml](configs/opensora-v1-1/train/stdit2_576x1024x24.yaml) |
| STDiT3-XL/2  |  8     | 1          | 64x576x1024  | ON         | bf16       | OFF        | O1           |  13 mins      | 18.94     | [yaml](configs/opensora-v1-1/train/stdit2_576x1024x24.yaml) |

> vae cache: whether vae embedding is pre-computed and cached before training.

Note that T5 text embedding is pre-computed before training.

Here are some generation results after fine-tuning STDiT2 on a mixkit subset.

<table class="center">
<tr>
  <td width=50% style="text-align:center;"><b>576x1024x48</b></td>
  <td width=50% style="text-align:center;"><b>576x1024x48</b></td>
  </tr>
<tr>
  <td width=50%><video src="https://github.com/mindspore-lab/mindone/assets/52945530/4df1dabf-1a7c-45d9-b005-08f6c2d26dfe" autoplay></td>
  <td width=50%><video src="https://github.com/mindspore-lab/mindone/assets/52945530/6e735171-042f-4b8d-a12c-4ddd5b2b4382" autoplay></td>
</tr>
<tr>
  <td width=50% style="text-align:center;"><b>576x1024x48</b></td>
  <td width=50% style="text-align:center;"><b>576x1024x48</b></td>
  </tr>
<tr>
  <td width=50%><video src="https://github.com/mindspore-lab/mindone/assets/52945530/ab627b2c-d932-4c9d-84f4-afe0c9d5d5ce" autoplay></td>
  <td width=50%><video src="https://github.com/mindspore-lab/mindone/assets/52945530/532f9d62-9b16-44dc-bd7a-4a24bd930e21" autoplay></td>
</tr>
</table>


</details>

### Open-Sora 1.0
<details>
<summary>View more</summary>

#### Training Performance

All experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.
| model name   | cards  | batch size | resolution   | stage | precision | sink |  jit level   | graph compile | s/step | recipe |
| :--:         | :--:   | :--:       | :--:         | :--:  | :--:      |:--:  | :--:         | :--:          |:--:    |:--:    |
| STDiT-XL/2  |  8     | 3          | 16x256x256   | 1     | fp16      |  ON  | O1           | 5~6 mins      |  1.53  | [yaml](configs/opensora/train/stdit_256x256x16_ms.yaml) |
| STDiT-XL/2  |  8     | 1          | 16x512x512   | 2     | fp16      |  ON  | O1           | 5~6 mins      |  2.47  | [yaml](configs/opensora/train/stdit_512x512x16.yaml) |
| STDiT-XL/2  |  8     | 1          | 64x512x512   | 3     | bf16      |  ON  | O1           | 5~6 mins      |  8.52  | [yaml](configs/opensora/train/stdit_512x512x64_ms.yaml) |


#### Loss Curves

<details>
<summary>Training loss curves </summary>

16x256x256 Pretraining Loss Curve:
![train_loss_256x256x16](https://github.com/SamitHuang/mindone/assets/8156835/c85ce7ce-a59c-4a5f-af40-a82a568ebd95)

16x256x256 HQ Training Loss Curve:
![train_loss_512x512x16](https://github.com/SamitHuang/mindone/assets/8156835/1926b12f-050d-47d2-bcec-fd08eef9f75b)

16x512x512 HQ Training Loss Curve:
![train_loss_512x512x64](https://github.com/SamitHuang/mindone/assets/8156835/d9287f36-888f-4ad6-92d9-5659eff0b306)

</details>


#### Text-to-Video Generation after Fine-tuning

Here are some generation results after fine-tuning STDiT on a subset of WebVid dataset.

<table class="center">
<tr>
  <td width=33% style="text-align:center;"><b>512x512x64</b></td>
  <td width=33% style="text-align:center;"><b>512x512x64</b></td>
  <td width=33% style="text-align:center;"><b>512x512x64</b></td>
</tr>
<tr>
  <td width=33%><video src="https://github.com/SamitHuang/mindone/assets/8156835/c82c059f-57da-44e5-933b-66ccf9e59ea0"></td>
  <td width=33%><video src="https://github.com/SamitHuang/mindone/assets/8156835/f00ad2bd-56e7-448c-9f85-c58888dca609"></td>
  <td width=33%><video src="https://github.com/SamitHuang/mindone/assets/8156835/51b4a431-195b-4a53-b177-e58a7aa7276c"></td>
</tr>
</table>


#### Quality Evaluation
For quality evaluation, please refer to the original HPC-AI Tech [evaluation doc](https://github.com/hpcaitech/Open-Sora/blob/main/eval/README.md) for video generation quality evaluation.

</details>

## VAE Training & Evaluation

A 3D-VAE pipeline consisting of a spatial VAE followed by a temporal VAE is trained in OpenSora v1.1. For more details, refer to [VAE Documentation](https://github.com/hpcaitech/Open-Sora/blob/main/docs/vae.md).

### Prepare Pretrained Weights

- Download pretained VAE-2D checkpoint from [PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae) if you aim to train VAE-3D from spatial VAE initialization.

    Convert to ms checkpoint:
    ```
    python tools/convert_vae1.2.py --src /path/to/pixart_sigma_sdxlvae_T5_diffusers/vae/diffusion_pytorch_model.safetensors --target models/sdxl_vae.ckpt --from_vae2d
    ```

- Downalod pretrained VAE-3D checkpoint from [hpcai-tech/OpenSora-VAE-v1.2](https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2/tree/main) if you aim to train VAEA-3D from the VAE-3D model pre-trained with 3 stages.

    Convert to ms checkpoint:
    ```
    python tools/convert_vae1.2.py --src /path/OpenSora-VAE-v1.2/models.safetensors --target models/OpenSora-VAE-v1.2/sdxl_vae.ckpt
    ```

- Download lpips mindspore checkpoint from [here](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) and put it under 'models/'


### Data Preprocess
Before VAE-3D training, we need to prepare a csv annotation file for the training videos. The csv file list the path to each video related to the root `video_folder`. An example is
```
video
dance/vid001.mp4
dance/vid002.mp4
...
```

Taking UCF-101 for example, please download the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset and extract it to `datasets/UCF-101` folder. You can generate the csv annotation by running  `python tools/annotate_vae_ucf101.py`. It will result in two csv files, `datasets/ucf101_train.csv` and `datasets/ucf101_test.csv`, for training and  testing respectively.


### Training
```bash
# stage 1 training, 8 NPUs
msrun --worker_num=8 --local_work_num=8 \
python scripts/train_vae.py --config configs/vae/train/stage1.yaml --use_parallel=True --csv_path datasets/ucf101_train.csv --video_folder datasets/UCF-101

# stage 2 training, 8 NPUs
msrun --worker_num=8 --local_work_num=8 \
python scripts/train_vae.py --config configs/vae/train/stage2.yaml --use_parallel=True --csv_path datasets/ucf101_train.csv --video_folder datasets/UCF-101

# stage 3 training, 8 NPUs
msrun --worker_num=8 --local_work_num=8 \
python scripts/train_vae.py --config configs/vae/train/stage3.yaml --use_parallel=True --csv_path datasets/ucf101_train.csv --video_folder datasets/UCF-101
```

You can change the `csv_path` and `video_folder` to train on your own data.

###  Performance Evaluation
To evaluate the VAE performance, you need to run VAE inference first to generate the videos, then calculate scores on the generated videos:

```bash
# video generation and evaluation
python scripts/inference_vae.py --ckpt_path /path/to/you_vae_ckpt --image_size 256 --num_frames=17 --csv_path datasets/ucf101_test.csv --video_folder datasets/UCF-101
```

You can change the `csv_path` and `video_folder` to evaluate on your own data.

Here, we report the training performance and evaluation results on the UCF-101 dataset.


All experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name   | cards  | batch size | resolution   |  precision  |  jit level    | graph compile | s/step    | PSNR   | SSIM  | recipe |
| :--:         | :--:   | :--:       | :--:         | :--:       | :--:         | :--:          | :--:      | :--:   | :--:      | :--:      |
| VAE-3D  |  8     | 1          | 17x256x256   | bf16       |  O1           |  5 mins      | 1.09     |  29.02   | 0.87  | [yaml](configs/vae/train/stage3.yaml) |


Note that we train with mixed video ang image strategy i.e. `--mixed_strategy=mixed_video_image` for stage 3 instead of random number of frames (`mixed_video_random`). Random frame training will be supported in the future.


## Long sequence training and inference (sequence parallel)

### Training

We support training with the OpenSora v1.2 model using SP (Sequence Parallel) and [DSP](https://arxiv.org/abs/2403.10266) (Dynamic Sequence Parallel), handling up to 408 frames (~16 seconds) on 4 NPU* cards. Additionally, we have optimized the training speed by implementing micro-batch parallelism in the VAE’s spatial and temporal domains, achieving approximately a 20% speed boost. We evaluate the training performance using the MixKit dataset, which includes high-resolution videos (1080P, duration 12s to 100s). The training performance results are reported below.

All experiments are tested on ascend 910* with mindspore 2.4.0 graph mode.
| model name   | cards  | batch size | resolution  | sink | precision   | jit level | graph compile |  s/step | recipe |
| :--:         | :--:   | :--:       | :--:       | :--:       | :--:      | :--:      |:--:          | :--:       | :--:   |
| STDiT3-XL/2  |  4     | 1          | 408x720x1280| OFF     |   bf16    | O1        |    12 mins   | 48.30   | [script](scripts/run/run_train_os1.2_stage2_sp.sh)
| STDiT3-XL/2  |  4     | 1          | 408x720x1280| OFF     |   bf16    | O1        |    12 mins   | 47.00   | [script](scripts/run/run_train_os1.2_stage2_dsp.sh)

> To prevent the system from running out of memory, ensure you launch the training job on a server with sufficient memory. For 4P training, at least 400GB of memory is required.


### Inference

We evaluate the inference performance of text-to-video generation by measuring the average sampling time per step and the total sampling time of a video.

All experiments are tested on ascend 910* with mindspore 2.4.0 graph mode.


| model name      |  cards | batch size | resolution |  precision | scheduler   |  steps   |  jit level |   graph compile | s/step     | s/video | recipe |
| :--:         | :--:   | :--:       | :--:       | :--:       | :--:       | :--:       | :--:       | :--:      |:--:    | :--:   |:--:   |
| STDiT3-XL/2  |  2     | 1          | 408x720x1280   |  bf16    |   RFlow   |   30   |   O0  | 1~2 mins |  26.03    |    780.00      |  [script](scripts/run/run_infer_sequence_parallel.sh) |
| STDiT3-XL/2  |  2     | 1          | 408x720x1280   |  bf16    |   RFlow   |   30   |   O0  | 1~2 mins |  22.03    |    660.00      |  [script](scripts/run/run_infer_sequence_parallel.sh) |


## Training and Inference Using the FiT-Like Pipeline

<details>
<summary>View more</summary>

> ⚠️**WARNING:** This feature is experimental. The official version is under development.

We provide support for training Open-Sora 1.1 using the FiT-Like pipeline as an alternative solution for handling multi-resolution videos, in contrast to the bucketing strategy.

### FiT-Like Training

To begin, we need to prepare the VAE (Variational Autoencoder) latents from multi-resolution videos. For instance, if you intend to train at a resolution of up to 512x512 pixels, please run

```bash
python script/infer_vae.py \
    --csv_path /path/to/video_caption.csv  \
    --video_folder /path/to/video_folder  \
    --output_path /path/to/video_embed_folder  \
    --vae_checkpoint models/sd-vae-ft-ema.ckpt \
    --image_size 512 \
    --resize_by_max_value True \
    --vae-micro-batch-size 1
    --mode 1
```

The extracted VAE latent will be saved in the video embedding folder.

Then, to launch a distributed training with eight NPU cards, please run

```bash
msrun --worker_num=8 --local_worker_num=8  \
    scripts/train.py --config configs/opensora-v1-1/train/train_stage1_fit.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video_folder \
    --text_embed_folder /path/to/text_embed_folder \
    --vae_latent_folder /path/to/video_embed_folder \
    --use_parallel True \
    --max_image_size 512 \
```

We evaluated the training performance on MindSpore and Ascend NPUs. The results are as follows.

| Model       | Context      | Precision | BS | NPUs | Max. Size | Train T. (s/step) |
|:------------|:-------------|:----------|:--:|:----:|:---------------:|:-----------------:|
| STDiT2-XL/2 | D910\*-MS2.3_master | BF16      | 1  |  4   | 16x512x512      |       2.3         |


### FiT-Like Inference

To sample a video with a resolution of 384x672 using the trained checkpoint. You can run

```bash
python scripts/inference_i2v.py --config configs/opensora-v1-1/inference/t2v_fit.yaml \
    --ckpt_path /path/to/your/opensora-v1-1.ckpt \
    --prompt_path /path/to/prompt.txt \
    --image_size 384 672 \
    --max_image_size 512 \
```

Make sure that the `max_image_size` parameter remains consistent between your training and inference commands.

Here are some generation results after fine-tuning STDiT on a small dataset:

<table class="center">
<tr>
  <td style="text-align:center;"><b>384x672x16</b></td>
  <td style="text-align:center;"><b>672x384x16</b></td>
</tr>
<tr>
  <td><video src="https://github.com/zhtmike/mindone/assets/8342575/97d8f37d-8ac3-49a8-af6d-5103f299e481" autoplay></td>
  <td><video src="https://github.com/zhtmike/mindone/assets/8342575/abefa666-8e88-4eef-974e-a4d4bfa1cd53" autoplay></td>
</tr>
</table>

</details>

## Contribution

Thanks go to the support from MindSpore team and the open-source contributions from the OpenSora project.

If you wish to contribute to this project, you can refer to the [Contribution Guideline](../../CONTRIBUTING.md).

## Acknowledgement

* [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization
  system.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT): An acceleration for DiT training. We adopt valuable acceleration
  strategies for training progress from OpenDiT.
* [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
* [Latte](https://github.com/Vchitect/Latte): An attempt to efficiently train DiT for video.
* [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
* [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
* [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
* [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful image captioning model based on [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [Yi-34B](https://huggingface.co/01-ai/Yi-34B).
* [DSP](https://github.com/NUS-HPC-AI-Lab/VideoSys): Dynamic Sequence Parallel introduced by NUS HPC AI Lab.

We are grateful for their exceptional work and generous contribution to open source.
