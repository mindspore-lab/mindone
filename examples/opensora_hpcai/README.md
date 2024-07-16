

## Open-Sora: Democratizing Efficient Video Production for All

Here we provide an efficient MindSpore implementation of [OpenSora](https://github.com/hpcaitech/Open-Sora), an open-source project that aims to foster innovation, creativity, and inclusivity within the field of content creation.

This repository is built on the models and code released by HPC-AI Tech. We are grateful for their exceptional work and generous contribution to open source.

<h4>Open-Sora is still at an early stage and under active development.</h4>



## 📰 News & States

| Official News from HPC-AI Tech                                                                                                                                                                                                                                                                                                                                                | MindSpore Support                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **[2024.04.25]** 🤗 HPC-AI Tech released the [Gradio demo for Open-Sora](https://huggingface.co/spaces/hpcai-tech/open-sora) on Hugging Face Spaces.                                                                                                                                                                                                                          | N.A.                                                                                           |
| **[2024.04.25]** 🔥 HPC-AI Tech released **Open-Sora 1.1**, which supports **2s~15s, 144p to 720p, any aspect ratio** text-to-image, **text-to-video, image-to-video, video-to-video, infinite time** generation. In addition, a full video processing pipeline is released. [[checkpoints]]() [[report]](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_02.md) | Image/Video-to-Video; Infinite time generation; Variable resolutions, aspect ratios, durations |
| **[2024.03.18]** HPC-AI Tech released **Open-Sora 1.0**, a fully open-source project for video generation.                                                                                                                                                                                                                                                                    | ✅ VAE + STDiT training and inference                                                           |
| **[2024.03.04]** HPC-AI Tech Open-Sora provides training with 46% cost reduction [[blog]](https://hpc-ai.com/blog/open-sora)                                                                                                                                                                                                                                                  | ✅ Parallel training on Ascend devices                                                          |



## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910*.

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

### TODO
* [ ] Optimizer-parallel and sequence-parallel training **[WIP]**
* [ ] Scaling model parameters and dataset size.

Your contributions are welcome.

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
* [Contribution](#contribution)
* [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

* Repo structure: [structure.md](docs/structure.md)


## Installation

1. Install MindSpore according to the [official instructions](https://www.mindspore.cn/install).
    For Ascend devices, please install **CANN driver C18 (0517)** from [here](https://repo.mindspore.cn/ascend/ascend910/20240517/) and install **MindSpore 2.3-master (0615)** from [here](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/).

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

### Open-Sora 1.1 Model Weights

- STDit:

| Stage | Resolution         | Model Size | Data                       | #iterations | URL                                                                    |
|-------|--------------------|------------|----------------------------|-------------|------------------------------------------------------------------------|
| 2     | mainly 144p & 240p | 700M       | 10M videos + 2M images     | 100k        | [Download](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage2) |
| 3     | 144p to 720p       | 700M       | 500K HQ videos + 1M images | 4k          | [Download](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v2-stage3) |

  Convert to ms checkpoint:

  ```
  python tools/convert_pt2ms.py --src /path/to/OpenSora-STDiT-v2-stage3/model.safetensors --target models/opensora_v1.1_stage3.ckpt
  ```

- T5 and VAE models are identical to OpenSora 1.0 and can be downloaded from the links below.

### Open-Sora 1.0 Model Weights

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


## Inference


### Open-Sora 1.1 Command Line Inference

#### Image/Video-to-Video Generation (supports text guidance)

```shell
python scripts/inference.py --config configs/opensora-v1-1/inference/sample_iv2v.yaml --ckpt_path /path/to/your/opensora-v1-1.ckpt
```
> for parallel inference, please use `mpirun` or `msrun`, and append `--use_parallel=True` to the inference script referring to `scripts/run/run_infer_os_v1.1_t2v_parallel.sh`

In the `sample_iv2v.yaml`, provide such information as `loop`, `condition_frame_length`, `captions`, `mask_strategy`,
and `reference_path`. See [here](docs/quick_start.md#imagevideo-to-video) for more details.

#### Text-to-Video Generation

To generate a video from text, you can use `sample_t2v.yaml` or set `--reference_path` to an empty string `''`
when using `sample_iv2v.yaml`.

```shell
python scripts/inference.py --config configs/opensora-v1-1/inference/sample_t2v.yaml --ckpt_path /path/to/your/opensora-v1-1.ckpt
```

### Open-Sora 1.0 Command Line Inference

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


## Data Processing

Currently, we didn't implement the complete pipeline for data processing from raw videos to high-quality text-video pairs. We provide the data processing tools as follows.

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
</details>


### Cache Text Embeddings

For acceleration, we pre-compute the t5 embedding before training stdit.

```bash
python scripts/infer_t5.py \
    --csv_path /path/to/video_caption.csv \
    --output_path /path/to/text_embed_folder \
    --model_max_length 200 # For OpenSora v1.1
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

### Open-Sora 1.1 Training

Stand-alone training for Stage 1 of OpenSora v1.1:

```shell
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
   in [Multi-resolution Training with Buckets](./docs/quick_start.md#4-multi-resolution-training-with-buckets-opensora-v11-only).


### Open-Sora 1.0 Training

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

To train in bfloat16 precision, please parse `--global_bf16=True`

For more usage, please check `python scripts/train.py -h`.
You may also see the example shell scripts in `scripts/run` for quick reference.


## Evaluation

### Open-Sora 1.1

#### Training Performance

We evaluated the training performance on MindSpore and Ascend NPUs. The results are as follows.

| Model       | Context      | jit_level | Precision | BS | NPUs | Resolution(framesxHxW) | Train T. (s/step) |
|:------------|:-------------|:--------|:---------:|:--:|:----:|:----------------------:|:-----------------:|
| STDiT2-XL/2 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/)  |    O1  |    BF16   |  1 |  8   |       16x512x512       |        2.00       |
| STDiT2-XL/2 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/)  |    O1  |    BF16   |  1 |  8   |       64x512x512       |        8.30       |
| STDiT2-XL/2 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) |    O1  |    BF16   |  1 |  8   |       24x576x1024      |        8.22       |
| STDiT2-XL/2 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) |    O1  |    BF16   |  1 |  8   |       24x576x1024      |        7.82       |
| STDiT2-XL/2 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) |    O1  |    BF16   |  1 |  8   |       64x576x1024      |        21.15      |
| STDiT2-XL/2 | D910\*-[CANN C18(0517)](https://repo.mindspore.cn/ascend/ascend910/20240517/)-[MS2.3_master(0615)](https://repo.mindspore.cn/mindspore/mindspore/version/202406/20240615/master_20240615020018_43ccb91e45899b64fe31d304497ab17e3ada3cea_newest/unified/) |    O1  |    BF16   |  1 |  8   |       24x1024x1024     |        16.98      |
> Context: {G:GPU, D:Ascend}{chip type}-{mindspore version}.

>Note that the above performance uses both t5 cached embedding data and vae cached latent data.

** Tips ** for performance optimization: to speed up training, you can set `dataset_sink_mode` as True and reduce `num_recompute_blocks` from 28 to a number that doesn't lead to out-of-memory.

Here are some generation results after fine-tuning STDiT2 on small dataset.

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


### Open-Sora 1.0

#### Training Performance

We evaluated the training performance on MindSpore and Ascend NPUs. The results are as follows.

| Model      | Context      | Precision | BS | NPUs | Resolution  | Train T. (s/step) |
|:-----------|:-------------|:----------|:--:|:----:|:-----------:|:-----------------:|
| STDiT-XL/2 | D910\*-MS2.3 | FP16      | 2  |  8   | 16x256x256  |       1.10        |
| STDiT-XL/2 | D910\*-MS2.3 | FP16      | 1  |  8   | 16x512x512  |       1.67        |
| STDiT-XL/2 | D910\*-MS2.3 | FP16      | 1  |  8   | 64x512x512  |       5.72        |
| STDiT-XL/2 | D910\*-MS2.3 | BF16      | 1  |  8   | 64x512x512  |       6.80        |
| STDiT-XL/2 | D910\*-MS2.3 | FP16      | 1  |  8   | 300x512x512 |        37         |
> Context: {G:GPU, D:Ascend}{chip type}-{mindspore version}.

Note that training on 300 frames at 512x512 resolution is achieved by optimization+data parallelism with t5 cached embeddings.

** Tips ** for performance optimization: to speed up training, you can set `dataset_sink_mode` as True and reduce `num_recompute_blocks` from 28 to a number that doesn't lead to out-of-memory.

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


## Training and Inference Using the FiT-Like Pipeline

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

| Model       | Context      | Precision | BS | NPUs | Max. Resolution | Train T. (s/step) |
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

Here are some generation results after fine-tuning STDiT on small dataset:

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

We are grateful for their exceptional work and generous contribution to open source.
