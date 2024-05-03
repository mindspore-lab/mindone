

## Open-Sora: Democratizing Efficient Video Production for All

Here we provide an efficient MindSpore implementation of [OpenSora](https://github.com/hpcaitech/Open-Sora), an open-source project that aim to foster innovation, creativity, and inclusivity within the field of content creation.

This repository is built on the models and code released by hpcaitech. We are grateful for their exceptional work and generous contribution to open source.

<h4>Open-Sora is still at an early stage and under active development.</h4>



## 📰 News & States

|        Official News from hpcaitech  | MindSpore Support     |
| ------------------ | ---------- |
| **[2024.04.25]** 🤗 hpcaitech released the [Gradio demo for Open-Sora](https://huggingface.co/spaces/hpcai-tech/open-sora) on Hugging Face Spaces. | N.A.      |
| **[2024.04.25]** 🔥 hpcaitech released **Open-Sora 1.1**, which supports **2s~15s, 144p to 720p, any aspect ratio** text-to-image, **text-to-video, image-to-video, video-to-video, infinite time** generation. In addition, a full video processing pipeline is released. [[checkpoints]]() [[report]](/docs/report_02.md) | Coming soon     |
| **[2024.03.18]** hpcaitech released **Open-Sora 1.0**, a fully open-source project for video generation.  |  ✅ VAE + STDiT training and inference |
| **[2024.03.04]** hpcaitech Open-Sora provides training with 46% cost reduction [[blog]](https://hpc-ai.com/blog/open-sora) | ✅ Parallel training on Ascend devices|



## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910*.

<summary>OpenSora 1.0 Demo</summary>

| **2s 512×512**                                                                                                                                                                 | **2s 512×512**                                                                                                                                                              | **2s 512×512**                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
|    ![009-A-serene-night-scene-in-a-forested-area -The-first](https://github.com/SamitHuang/mindone/assets/8156835/72f0dd45-bcf5-47b2-b2b3-24599bd9b16e) |      ![000-A-soaring-drone-footage-captures-the-majestic-beauty-of-a](https://github.com/SamitHuang/mindone/assets/8156835/6bde280b-80a7-4617-a53d-58981ef308c2) |  ![001-A-majestic-beauty-of-a-waterfall-cascading-down-a-cliff](https://github.com/SamitHuang/mindone/assets/8156835/a0b5d303-71d7-4de0-9592-0784bac398bf)|
| A serene night scene in a forested area. [...] The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. | A soaring drone footage captures the majestic beauty of a coastal cliff, [...] The water gently laps at the rock base and the greenery that clings to the top of the cliff. | The majestic beauty of a waterfall cascading down a cliff into a serene lake. [...] The camera angle provides a bird's eye view of the waterfall. |
|   ![006-A-bustling-city-street-at-night,-filled-with-the-glow](https://github.com/SamitHuang/mindone/assets/8156835/00a966c8-16fa-4799-98a6-3d69c2983e49) |        ![002-A-vibrant-scene-of-a-snowy-mountain-landscape -The-sky](https://github.com/SamitHuang/mindone/assets/8156835/fb243b36-b2dd-4bac-a8b2-812b5c3b35da)  |  ![004-A-serene-underwater-scene-featuring-a-sea-turtle-swimming-through](https://github.com/SamitHuang/mindone/assets/8156835/31a7f201-b436-4a85-a68c-e0cd58d8bca5) |
| A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. [...]                                                           | The vibrant beauty of a sunflower field. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. [...]                                            | A serene underwater scene featuring a sea turtle swimming through a coral reef. The turtle, with its greenish-brown shell [...]    |

Videos are downsampled to `.gif` for display. Click for original videos. Prompts are trimmed for display, see [here](/assets/texts/t2v_samples.txt) for full prompts.


## 🔆 Features

- 📍 **Open-Sora 1.0** with the following features
    - ✅ Text-to-video generation in 256x256 or 512x512 resolution and up to 64 frames.
    - ✅ Three-stage training: i) 16x256x256 video pretraining, ii) 16x512x512 video finetuning, and iii) 64x512x512 videos
    - ✅ Optimized training receipes for MindSpore+Ascend framework (see `configs/opensora/train/xxx_ms.yaml`)
    - ✅ Acceleration methods: flash attention, recompute (graident checkpointing), data sink, mixed precision, and graph compilation.
    - ✅ Data parallelism + Optimizer parallelism, allow training on 300x512x512 videos

<details>
<summary>View more</summary>

* ✅ Following the findings in OpenSora, we also adopt the VAE from stable diffusion for video latent encoding.
* ✅ We pick the **STDiT** model as our video diffusion transformer following the best practice in OpenSora.
* ✅ Support T5 text conditioning.

</details>

### TODO
* [ ] Support OpenSora 1.1 **[WIP]**
    - [ ] Support variable aspect ratios, resolutions, and durations.
    - [ ] Support image and video conditioning
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

1. Install MindSpore 2.3rc1 according to the [official instruction](https://www.mindspore.cn/install)
> To use flash attention, it's recommended to use mindspore 2.3rc2 (release soon).


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

Coming soon.

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
    python tools/convert_vae.py --source /path/to/sd-vae-ft-ema/diffusion_pytorch_model.safetensors --target models/sd-vae-ft-ema.ckpt
    ```

- STDiT: Download `OpenSora-v1-16x256x256.pth` / `OpenSora-v1-HQ-16x256x256.pth` / `OpenSora-v1-HQ-16x512x512.pth` from [here](https://huggingface.co/hpcai-tech/Open-Sora/tree/main)

    Convert to ms checkpoint:

    ```
    python tools/convert_pt2ms.py --src /path/to/OpenSora-v1-16x256x256.pth --target models/OpenSora-v1-16x256x256.ckpt
    ```

    Training orders: 16x256x256 $\rightarrow$ 16x256x256 HQ $\rightarrow$ 16x512x512 HQ.

    These model weights are partially initialized from [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha). The number of
parameters is 724M. More information about training can be found in hpcaitech's **[report](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md)**. More about the dataset can be found in [datasets.md](https://github.com/hpcaitech/Open-Sora/blob/main/docs/datasets.md) from hpcaitech. HQ means high quality.

- PixArt-α: Download the pth checkpoint from [here](https://download.openxlab.org.cn/models/PixArt-alpha/PixArt-alpha/weight/PixArt-XL-2-512x512.pth)  (for training only)

    Convert to ms checkpoint:
    ```
    python tools/convert_pt2ms.py --src /path/to/PixArt-XL-2-512x512.pth --target models/PixArt-XL-2-512x512.ckpt
    ```


## Inference


### Open-Sora 1.1 Command Line Inference

Coming soon

### Open-Sora 1.0 Command Line Inference

You can run text-to-video inference via the script `scripts/inference.py` as follows.

```bash
# Sample 16x256x256 videos
python scripts/inference.py --config configs/opensora/inference/stdit_256x256x16.yaml --ckpt_path models/OpenSora-v1-16x256x256.ckpt --prompt_path /path/to/prompt.txt

# Sample 16x512x512 videos
python scripts/inference.py --config configs/opensora/inference/stdit_512x512x16.yaml --ckpt_path models/OpenSora-v1-16x512x512.ckpt --prompt_path /path/to/prompt.txt

# Sample 64x512x512 videos
python scripts/inference.py --config configs/opensora/inference/stdit_512x512x64.yaml --ckpt_path /path/to/your/opensora.ckpt --prompt_path /path/to/prompt.txt
```

We also provide a three-stage sampling script `run_sample_3stages.sh` to reduce memory limitation, which decomposes the whole pipeline into text embedding, text-to-video latent sampling, and vae decoding.

For more usage on the inference script, please run `python scripts/inference.py -h`


## Data Processing

Currently, we didn't implement the complete pipeline for data processing from raw videos to high-quality text-video pairs. We provide the data processing tools as follows.

<details>
<summary>View more</summary>

The text-video pair data should be organized as follows for example.

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
video_folder/part01/vid001.mp4,a cartoon character is walking throu
video_folder/part01/vid002.mp4,a red and white ball with an angry look on its face
```

### Cache Text Embeddings

For acceleration, we pre-compute the t5 embedding before training stdit.

```bash
python scripts/infer_t5.py \
    --csv_path /path/to/video_caption.csv \
    --output_dir /path/to/text_embed_folder \
```

After running, the text embeddings saved as npz file for each caption will be in `output_dir`. Please change `csv_path` to your video-caption annotation file accordingly.

### Cache Video Embedding (Optional)

If the storage budget is sufficient, you may also cache the video embedding by

```bash
python scripts/infer_vae.py\
    --csv_path /path/to/video_caption.csv  \
    --video_folder /path/to/video_folder  \
    --output_dir /path/to/video_embed_folder  \
    --vae_checkpoint models/sd-vae-ft-ema.ckpt \
    --image_size 512 \
```

After running, the vae latents saved as npz file for each video will be in `output_dir`.

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

</details>

## Training

### Open-Sora 1.1 Training

Coming soon

### Open-Sora 1.0 Training

Once the training data including the [T5 text embeddings](#cache-text-embeddings) is prepared, you can run the following commands to launch training.

```bash
# standalone training, 16x256x256
python scripts/train.py --config configs/opensora/train/stdit_256x256x16.yaml \
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

For more usages, please check `python scripts/train.py -h`. You may also see the example shell scripts in `scripts/run` for quick reference.

#### Performance

We evaluated the training performance on MindSpore and Ascend NPUs. The results are as follows.

| Model          |   Context      |  Precision         |  BS  | NPUs  |   Resolution  |  Train T. (s/step)  |
|:---------------|:---------------|:--------------|:-------------:|:---------:|:----------:|:------------:|
| STDiT-XL/2     |    D910\*x1-MS2.3       |      FP16   |      2 | 8    |    16x256x256  |  1.10  |
| STDiT-XL/2     |    D910\*x1-MS2.3       |      FP16   |      1 | 8    |    16x512x512  |  1.67   |
| STDiT-XL/2     |    D910\*x1-MS2.3       |      FP16   |      1 | 8    |    64x512x512  |  6.70  |
| STDiT-XL/2     |    D910\*x1-MS2.3       |      FP16   |      1 | 64    |    64x512x512  | 6.81  |
| STDiT-XL/2     |    D910\*x1-MS2.3       |      FP16   |      1 | 8    |    300x512x512  | 37  |
> Context: {G:GPU, D:Ascend}{chip type}-{number of NPUs}-{mindspore version}.

Note that training on 300 frames at 512x512 resolution are achived by optimization+data parallelism with t5 cached embeddings.


## Evaluation

Please refer to the original hpcaitech [evaluation doc](https://github.com/hpcaitech/Open-Sora/blob/main/eval/README.md)

## Contribution

Thanks goes to the support from MindSpore team and the open-source contributions from the OpenSora project.

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
