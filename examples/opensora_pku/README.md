# Open-Sora Plan

Here we provide an efficient MindSpore version of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main) from Peking University. We would like to express our gratitude to their contributions! :+1:

**OpenSora-PKU is still at an early stage and under active development.** Currently, we are in line with **Open-Sora-Plan v1.0.0**.

## 📰 News & States

|        Official News from OpenSora-PKU  | MindSpore Support     |
| ------------------ | ---------- |
| **[2024.04.09]** 🚀 PKU shared the latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), and the dataset for train (updating): [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset).| N.A.  |
| **[2024.04.07]** 🔥🔥🔥 PKU released Open-Sora-Plan v1.0.0. See their [report]([docs/Report-v1.0.0.md](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md)). | ✅ CausalVAE+LatteT2V+T5 inference and three-stage training (`17x256x256`, `65x256x256`, `65x512x512`)  |
| **[2024.03.27]** 🚀🚀🚀 PKU released the report of [VideoCausalVAE](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Train_And_Eval_CausalVideoVAE.md), which supports both images and videos.  | ✅ CausalVAE training and inference |
| **[2024.03.10]** 🚀🚀🚀 PKU supports training a latent size of 225×90×90 (t×h×w), which means to **train 1 minute of 1080P video with 30FPS** (2× interpolated frames and 2× super resolution) under class-condition.| frame interpolation and super-resolution are under-development.|
| **[2024.03.08]** PKU support the training code of text condition with 16 frames of 512x512. |   ✅ CausalVAE+LatteT2V+T5 training (`16x512x512`)|
| **[2024.03.07]** PKU support training with 128 frames (when sample rate = 3, which is about 13 seconds) of 256x256, or 64 frames (which is about 6 seconds) of 512x512. | class-conditioned training is under-development.|

## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910*.

<summary>Open-Sora-Plan v1.0.0 Demo</summary>

| 65×512×512 (2.7s) | 65×512×512 (2.7s) | 65×512×512 (2.7s) |
| --- | --- | --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/t2v/0-A%20serene%20underwater%20scene%20featuring%20a%20sea%20turtle%20swimming%20through%20a%20coral%20reef.%20The%20turtle,%20with%20its.gif?raw=true" width=224> | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/t2v/0-Yellow%20and%20black%20tropical%20fish%20dart%20through%20the%20sea.gif?raw=true" width=224>  | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/t2v/0-a%20dynamic%20interaction%20between%20the%20ocean%20and%20a%20large%20rock.%20The%20rock,%20with%20its%20rough%20texture%20and%20jagge.gif?raw=true" width=224> |
| A serene underwater scene featuring a sea turtle swimming... | Yellow and black tropical fish dart through the sea.  | a dynamic interaction between the ocean and a large rock...  |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/t2v/0-The%20dynamic%20movement%20of%20tall,%20wispy%20grasses%20swaying%20in%20the%20wind.%20The%20sky%20above%20is%20filled%20with%20clouds.gif?raw=true" width=224> |<img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/t2v/0-Slow%20pan%20upward%20of%20blazing%20oak%20fire%20in%20an%20indoor%20fireplace.gif?raw=true" width=224> | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/t2v/0-A%20serene%20waterfall%20cascading%20down%20moss-covered%20rocks,%20its%20soothing%20sound%20creating%20a%20harmonious%20symph.gif?raw=true" width=224>  |
| The dynamic movement of tall, wispy grasses swaying in the wind... | Slow pan upward of blazing oak fire in an indoor fireplace.  | A serene waterfall cascading down moss-covered rocks...  |


Videos are saved to `.gif` for display. See the text prompts in `examples/prompt_list_0.txt`.

## 🔆 Features

- 📍 **Open-Sora-Plan v1.0.0** with the following features
    - ✅ CausalVAE-4x8x8 training and inference. Supports video reconstruction.
    - ✅ T5 TextEncoder model inference.
    - ✅ Text-to-video generation in 256x256 or 512x512 resolution and up to 65 frames.
    - ✅ Three-stage training: i) 17x256x256 pretraining, ii) 65x256x256 finetuning, and iii) 65x512x512 finetuning.
    - ✅ Acceleration methods: flash attention, recompute (graident checkpointing), mixed precision, data parallelism, etc..


### TODO
* [ ] Optimizer-parallel and sequence-parallel training **[WIP]**
* [ ] Scaling model parameters and dataset size.
* [ ] Evaluation of various metrics.

You contributions are welcome.

<details>
<summary>View more</summary>

* [ ] Super-resolution model
* [ ] frame-interpolation model
</details>

## Contents

* [Installation](#installation)
* [Model Weights](#model-weights)
* [Inference](#inference)
* [Data Processing](#data-processing)
* [Training](#training)
* [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

* Repo structure: [structure.md](docs/structure.md)


## Installation
1. Use python>=3.8 [[install]](https://www.python.org/downloads/)

2. Install MindSpore 2.3rc1 according to the [official instruction](https://www.mindspore.cn/install)
> To use flash attention, it's recommended to use mindspore 2.3rc2 (release soon).


3. Install requirements
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

### Open-Sora-Plan v1.0.0 Model Weights

Please download the torch checkpoint of T5 from [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl), and download the opensora v1.0.0 models' weights from [LanguageBind/Open-Sora-Plan-v1.0.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main). Place them under `examples/opensora_pku` as shown below:
```bash
opensora_pku
├───LanguageBind
│   └───Open-Sora-Plan-v1.0.0
│       ├───17x256x256
│       │   ├───config.json
│       │   └───diffusion_pytorch_model.safetensors
│       ├───65x256x256
│       │   ├───config.json
│       │   └───diffusion_pytorch_model.safetensors
│       ├───65x512x512
│       │   ├───config.json
│       │   └───diffusion_pytorch_model.safetensors
│       └───vae
│          ├───config.json
│          └───diffusion_pytorch_model.safetensors
└───DeepFloyd/
    └───t5-v1_1-xxl
        ├───config.json
        ├───pytorch_model-00001-of-00002.bin
        ├───pytorch_model-00002-of-00002.bin
        ├───pytorch_model.bin.index.json
        ├───special_tokens_map.json
        ├───spiece.model
        └───tokenizer_config.json
```

After all weights being downloaded, please run the following script to run model conversion.
```bash
bash scripts/model_conversion/convert_all.sh
```


## Inference

### CausalVAE Command Line Inference

You can run video-to-video reconstruction task using `scripts/causalvae/reconstruction.sh`:
```bash
python examples/rec_imvi_vae.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0/vae \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --device Ascend \
    --sample_rate 1 \
    --num_frames 65 \
    --resolution 512 \
    --crop_size 512 \
    --ae CausalVAEModel_4x8x8 \
    --enable_tiling \
```
Please change the `--video_path` to the existing video file path and `--rec_path` to the reconstructed video file path. You can set `--grid` to save the original video and the reconstructed video in the same output file.

You can also run video reconstruction given an input video folder. See `scripts/causalvae/gen_video.sh`.

Some reconstruction results are listed below (left: source video clip, right: reconstructed). As mentioned in the [OpenSora-PKU report-v1.0.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md#causalvideovae-1), the current released version of CausalVideoVAE (v1.0.0) still has two main drawbacks: motion blurring and gridding effect.

<p float="center">
<img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/causalvae/reconstruction/girl.gif?raw=true" width="50%" /><img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/causalvae/reconstruction/highway.gif?raw=true" width="50%" />
</p>

<p float="center">
<img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/causalvae/reconstruction/parrot.gif?raw=true" width="50%" /><img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/causalvae/reconstruction/waves.gif?raw=true" width="50%" />
</p>

### Open-Sora-Plan v1.0.0 Command Line Inference

You can run text-to-video inference using the script `scripts/text_condition/sample_video.sh`.
```bash
python opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --version 65x512x512 \
    --save_img_path "./sample_videos/prompt_list_0" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 250 \
    --enable_tiling
```
You can change the `version` to `17x256x256` or `65x256x256` to change the number of frames and resolutions.

> In case of OOM error, there are two options:
> 1. Pass `--enable_time_chunk True` to allow vae decoding temporal frames as small, overlapped chunks. This can reduce the memory usage, which sacrificies a bit of temporal consistency.
> 2. Seperate the inference into two stages. In stage 1, please run inference with `--save_latents`. This will save some `.npy` files in the output directory. Then in stage 2, please run the same inference script with `--decode_latents`. The generated videos will be saved in the output directory.


## Training

### Causal Video VAE

#### Preparation

To train the causal vae model, you need to prepare a video dataset. You can download this video dataset following the instruction of [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset).

Causal video vae can be initialized from vae 2d for better convergence. This can be done by inflating the 2d vae model checkpoint as follows:

```
python tools/model_conversion/inflate_vae2d_to_vae3d.py \
    --src /path/to/vae_2d.ckpt  \
    --target pretrained/causal_vae_488_init.ckpt
```
> In case you lack vae 2d checkpoint in mindspore format, please use `tools/model_conversion/convert_vae.py` for model conversion, e.g. after downloading the [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) weights.

Please also download [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) and put it under `pretrained/` for training with lpips loss.

#### Standalone Training

To launch a single-card training, you can refer to `scripts/causalvae/train.sh`. Please revise the `--video_path` to the path of the folder where the videos are stored, and run:
```bash
bash scripts/causalvae/train.sh
```

#### Multi-Device Training

For parallel training, please use `msrun` and pass `--use_parallel=True`.
```bash
# 8 NPUs
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="output_log"  \
    python opensora/train/train_causalvae.py  \
    --use_parallel True \
    ... # pass other arguments
```

### Training Diffusion Model

#### Preparation

The first-stage training depends on the `t2v.pt` from [Vchitect/Latte](https://huggingface.co/maxin-cn/Latte/tree/main). Please download `t2v.pt` and place it under `pretrained/t2v.pt`. Then run model conversion with:
```bash
python tools/model_conversion/convert_latte.py \
  --src pretrained/t2v.pt \
  --target pretrained/t2v.ckpt
```

The [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset) includes the video files and one json file which records the video paths and captions. Please pass the json file path to `opensora/train/train_t2v.py` via `--data_path` and pass the video folder path to `opensora/train/train_t2v.py` via `--video_folder`.

For acceleration, we pre-compute the t5 embedding before training the diffusion transformer.

```bash
python opensora/sample/sample_text_embed.py \
    --data_file_path /path/to/video_caption.json \
    --output_dir /path/to/text_embed_folder \
```

After running, the text embeddings saved as npz file for each caption will be in `output_dir`. Please change `data_file_path` to your video-caption annotation file accordingly.

### Standalone Training

Before launching the first-stage training, please make sure the pretrained checkpoint is stored as `pretrained/t2v.ckpt`, and `--text_embed_folder` in the following shell scripts are set to the text embedding folder that you generated ahead.

```bash
# start 17x256x256 pretraining
bash scripts/text_condition/train_videoae_17x256x256.sh
```
After the first-stage training, please revise `scripts/text_condition/train_videoae_65x256x256.sh`, and change `--pretrained` to the checkpoint path from the last stage. Then run:

```bash
bash scripts/text_condition/train_videoae_65x256x256.sh
```
Simiarly, please revise the `--pretrained` checkpoint path and start the third-stage training with:

```bash
# start 65x512x512 finetuning
bash scripts/text_condition/train_videoae_65x512x512.sh
```


### Multi-Device Training

For parallel training, please use `msrun` and pass `--use_parallel=True`.

```bash
# 8 NPUs, 64x512x512
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="output_log"  \
    python opensora/train/train_t2v.py  \
    --use_parallel True \
    ... # pass other arguments
```


## 👍 Acknowledgement
* [Latte](https://github.com/Vchitect/Latte): The **main codebase** we built upon and it is an wonderful video generated model.
* [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT): Video Generation using VQ-VAE and Transformers.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [FiT](https://github.com/whlzy/FiT): Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595): Extending Context Window of Large Language Models via Positional Interpolation.
