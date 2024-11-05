# Open-Sora Plan

Here we provide an efficient MindSpore version of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main) from Peking University. We would like to express our gratitude to their contributions! :+1:

**OpenSora-PKU is still under active development.** Currently, we are in line with **Open-Sora-Plan v1.2.0** ([commit id](https://github.com/PKU-YuanGroup/Open-Sora-Plan/commit/294993ca78bf65dec1c3b6fb25541432c545eda9)).

## 📰 News & States

|        Official News from OpenSora-PKU  | MindSpore Support     |
| ------------------ | ---------- |
| **[2024.10.16]** 🎉 PKU released version 1.3.0, featuring: **WFVAE**, **pompt refiner**, **data filtering strategy**, **sparse attention**, and **bucket training strategy**. They also support 93x480p within **24G VRAM**. More details can be found at their latest [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.3.0.md). | 📝 Working in Progress |
| **[2024.07.24]** 🔥🔥🔥 PKU launched Open-Sora Plan v1.2.0, utilizing a 3D full attention architecture instead of 2+1D. See their latest [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.2.0.md). | ✅ V.1.2.0 CausalVAE inference & OpenSoraT2V multi-stage training|
| **[2024.05.27]** 🚀🚀🚀 PKU launched Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out their latest [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.1.0.md). | ✅ V.1.1.0 CausalVAE inference and LatteT2V infernece & three-stage training (`65x512x512`, `221x512x512`, `513x512x512`) |
| **[2024.04.09]** 🚀 PKU shared the latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), and the dataset for train (updating): [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset).| N.A.  |
| **[2024.04.07]** 🔥🔥🔥 PKU released Open-Sora-Plan v1.0.0. See their [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md). | ✅ CausalVAE+LatteT2V+T5 inference and three-stage training (`17×256×256`, `65×256×256`, `65x512x512`)  |
| **[2024.03.27]** 🚀🚀🚀 PKU released the report of [VideoCausalVAE](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Train_And_Eval_CausalVideoVAE.md), which supports both images and videos.  | ✅ CausalVAE training and inference |
| **[2024.03.10]** 🚀🚀🚀 PKU supports training a latent size of 225×90×90 (t×h×w), which means to **train 1 minute of 1080P video with 30FPS** (2× interpolated frames and 2× super resolution) under class-condition.| frame interpolation and super-resolution are under-development.|
| **[2024.03.08]** PKU support the training code of text condition with 16 frames of 512x512. |   ✅ CausalVAE+LatteT2V+T5 training (`16x512x512`)|
| **[2024.03.07]** PKU support training with 128 frames (when sample rate = 3, which is about 13 seconds) of 256x256, or 64 frames (which is about 6 seconds) of 512x512. | class-conditioned training is under-development.|


## Requirements

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:       |   :---:         | :---:      | :---:                |
| 2.3.1     |  24.1RC2      |7.3.0.1.231|   8.0.RC2.beta1   |

## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910*.

<summary>Open-Sora-Plan v1.2.0 Demo</summary>

29×1280×720 Text-to-Video Generation.

| 29x720x1280 (1.2s) |
| --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.2/29x720p/0-A%20close-up%20of%20a%20woman%E2%80%99s%20face%2C%20illuminated%20by%20the%20soft%20light%20of%20dawn%2C%20her%20expression%20serene%20and%20conte.gif?raw=true" width=720> |
| A close-up of a woman’s face, illuminated by the soft light of dawn... |

| 29x720x1280 (1.2s) |
| --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.2/29x720p/0-A%20young%20man%20at%20his%2020s%20is%20sitting%20on%20a%20piece%20of%20cloud%20in%20the%20sky%2C%20reading%20a%20book..gif?raw=true" width=720>  |
| 0-A young man at his 20s is sitting on a piece of cloud in the sky, reading a book...  |

| 29x720x1280 (1.2s) |
| --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.2/29x720p/0-A%20close-up%20of%20a%20woman%20with%20a%20vintage%20hairstyle%20and%20bright%20red%20lipstick%2C%20gazing%20seductively%20into%20the%20.gif?raw=true" width=720> |
| 0-A close-up of a woman with a vintage hairstyle and bright red lipstick...  |

Videos are saved to `.gif` for display.

## 🔆 Features

- 📍 **Open-Sora-Plan v1.2.0** with the following features
    - ✅ CausalVAEModel_D4_4x8x8 inference. Supports video reconstruction.
    - ✅ mT5-xxl TextEncoder model inference.
    - ✅ Text-to-video generation up to 93 frames and 720x1280 resolution.
    - ✅ Multi-stage training using Zero2 and Sequence parallelism.
    - ✅ Acceleration methods: flash attention, recompute (graident checkpointing), mixed precision, data parallelism, optimizer-parallel, etc..
    - ✅ Evaluation metrics : PSNR and SSIM.


### TODO
* [ ] Image-to-Video model **[WIP]**.
* [ ] Scaling model parameters and dataset size **[WIP]**.
* [ ] Evaluation of various metrics **[WIP]**.

You contributions are welcome.

## Contents

* [Installation](#installation)
* [Model Weights](#model-weights)
* [Inference](#inference)
* [Training](#training)
* [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

## Installation
1. Use python>=3.8 [[install]](https://www.python.org/downloads/)

2. Please install MindSpore 2.3.1 according to the [MindSpore official website](https://www.mindspore.cn/install/) and install [CANN 8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) as recommended by the official installation website.


3. Install requirements
```bash
pip install -r requirements.txt
```

In case `decord` package is not available, try `pip install eva-decord`.
For EulerOS, instructions on ffmpeg and decord installation are as follows.

<details onclose>
<summary>How to install ffmpeg and decord</summary>

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

### Open-Sora-Plan v1.2.0 Model Weights

Please download the torch checkpoint of mT5-xxl from [google/mt5-xxl](https://huggingface.co/google/mt5-xxl/tree/main), and download the opensora v1.2.0 models' weights from [LanguageBind/Open-Sora-Plan-v1.2.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main). Place them under `examples/opensora_pku` as shown below:
```bash
mindone/examples/opensora_pku
├───LanguageBind
│   └───Open-Sora-Plan-v1.2.0
│       ├───1x480p/
│       ├───29x480p/
│       ├───29x720p/
│       ├───93x480p/
│       ├───93x480p_i2v/
│       ├───93x720p/
│       └───vae/
└───google/
    └───mt5-xxl/
        ├───config.json
        ├───generation_config.json
        ├───pytorch_model.bin
        ├───special_tokens_map.json
        ├───spiece.model
        └───tokenizer_config.json
```

Currently, we can load `.safetensors` files directly in MindSpore, but not `.bin` or `.ckpt` files. We recommend you to convert the
`vae/checkpoint.ckpt` and `mt5-xxl/pytorch_model.bin` files to `.safetensor` files manually by running the following commands:
```shell
python tools/model_conversion/convert_pytorch_ckpt_to_safetensors.py --src LanguageBind/Open-Sora-Plan-v1.2.0/vae/checkpoint.ckpt --target LanguageBind/Open-Sora-Plan-v1.2.0/vae/diffusion_pytorch_model.safetensors  --config LanguageBind/Open-Sora-Plan-v1.2.0/vae/config.json

python tools/model_conversion/convert_pytorch_ckpt_to_safetensors.py --src google/mt5-xxl/pytorch_model.bin --target google/mt5-xxl/model.safetensors  --config google/mt5-xxl/config.json
```

Once the checkpoint files have all been prepared, you can refer to the inference guidance below.

## Inference

### CausalVAE Command Line Inference

You can run video-to-video reconstruction task using `scripts/causalvae/rec_video.sh`:
```bash
python examples/rec_video.py \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --device Ascend \
    --sample_rate 1 \
    --num_frames 65 \
    --height 480 \
    --width 640 \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory
```
Please change the `--video_path` to the existing video file path and `--rec_path` to the reconstructed video file path. You can set `--grid` to save the original video and the reconstructed video in the same output file.

You can also run video reconstruction given an input video folder. See `scripts/causalvae/rec_video_folder.sh`.

### Open-Sora-Plan v1.2.0 Command Line Inference

You can run text-to-video inference on a single Ascend device using the script `scripts/text_condition/single-device/sample_t2v_29x720p.sh`.
```bash
python opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.2.0/29x720p \
    --num_frames 29 \
    --height 720 \
    --width 1280 \
    --cache_dir "./" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_D4_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae\
    --save_img_path "./sample_videos/prompt_list_0_29x720p" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --model_type "dit" \
```
You can change the `num_frames`, `height` and `width` to match with the training shape of different checkpoints, e.g., `29x480p` requires `num_frames=29`, `height=480` and `width=640`. In case of oom on your device, you can try to append `--save_memory` to the command above, which enables a more radical tiling strategy for causal vae.


If you want to run a multi-device inference, e.g., 8 cards, please use `msrun` and pass `--use_parallel=True` as the example below:

```bash
# 8 NPUs
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="output_log"  \
    python opensora/sample/sample_t2v.py \
    --use_parallel True \
    ... # pass other arguments
```

The command above will run a 8-card inference and save the log files into "output_log". `--master_port` specifies the scheduler binding port number. `--worker_num` and `--local_worker_num` should be the same to the number of running devices, e.g., 8.

In case of the following error:
```bash
RuntimtError: Failed to register the compute graph node: 0. Reason: Repeated registration node: 0
```

Please edit the `master_port` to a different port number in the range 1024 to 65535, and run the script again.

See more examples of multi-device inference scripts under `scripts/text_condifion/multi-devices`.


### Sequence Parallelism

We support running inference with sequence parallelism. Please see the `sample_t2v_29x480p_sp.sh` and `sample_t2v_29x720p_sp.sh` under `scripts/text_condition/multi-devices/`.

If you set `--sp_size 8` to run sequence parallelism on 8 NPUs, you should also edit as follows:
```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```
## Training

### Causal Video VAE

#### Preparation

**Step 1: Downloading Datasets**:

To train the causal vae model, you need to prepare a video dataset. Open-Sora-Plan-v1.2.0 trains vae in two stages. In the first stage, the authors trained vae on the Kinetic400 video dataset. Please download K400 dataset from [this repository](https://github.com/cvdfoundation/kinetics-dataset). In the second stage, they trained vae on Open-Sora-Dataset-v1.1.0 dataset. We give a tutorial on how to download the v1.1.0 datasets. See [downloading tutorial](./tools/download/README.md).

**Step 2: Converting Pretrained Weights**:

As with v1.1.0, they initialized from the [SD2.1 VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse) using tail initialization for better convergence. Please download the torch weight file from the given [URL](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main).

After downloading the [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) weights, you can run:
```bash
python tools/model_conversion/convert_vae_2d.py --src path/to/diffusion.safetensor --target /path/to/sd-vae-ft-mse.ckpt.
```
This can convert the torch weight file into mindspore weight file.

They you can inflate the 2d vae model checkpoint into a 3d causal vae initial weight file as follows:

```bash
python tools/model_conversion/inflate_vae2d_to_vae3d.py \
    --src /path/to/sd-vae-ft-mse.ckpt  \
    --target pretrained/causal_vae_488_init.ckpt
```

In order to train vae with lpips loss, please also download [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) and put it under `pretrained/`.

#### Standalone Training

The first-stage training is conducted on 25-frame 256×256 videos of the [K400](https://github.com/cvdfoundation/kinetics-dataset) dataset. you can revise the `--video_path` in the training script to the video folder path of your downloaded dataset. This will allow the training script to load all video files under `video_path` in a **recursive manner**, and use them as the training data. Make sure the `--load_from_checkpoint` is set to the pretrained weight, e.g., `pretrained/causal_vae_488_init.ckpt`.


<details>
<summary>How to define the train and test set?</summary>

If you need to define the video files in the training set, please use a csv file with only one column, like:
```csv
"video"
folder_name/video_name_01.mp4
folder_name/video_name_02.mp4
...
```
Afterwards, you should revise the training script as below:
```bash
python opensora/train/train_causalvae.py \
    --data_file_path path/to/train_set/csv/file \
    --video_column "video" \
    --video_path path/to/downloaded/dataset \
    # pass other arguments
```

Similarly, you can create a csv file to include the test set videos, and pass the csv file to `--data_file_path` in `examples/rec_video_vae.py`.
</details>

To launch a single-card training, please run:
```bash
bash scripts/causalvae/train_with_gan_loss.sh
```

> Note:
> - Supports resume training by setting `--resume_training_checkpoint True`. It is the same for the multi-device training script.

#### Multi-Device Training

For parallel training, please use `msrun` and pass `--use_parallel=True`.
```bash
# 8 NPUs
msrun --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="output_log"  \
    python opensora/train/train_causalvae.py  \
    --use_parallel True \
    ... # pass other arguments, please refer to the single-device training script.
```
For more details, please take `scripts/causalvae/train_with_gan_loss_multi_device.sh` as an example.


#### Inference After Training

After training, you will find the checkpoint files under the `ckpt/` folder of the output directory. To evaluate the reconstruction of the checkpoint file, you can take `scripts/causalvae/rec_video_folder.sh` and revise it like:

```bash
python examples/rec_video_folder.py \
    --batch_size 1 \
    --real_video_dir input_real_video_dir \
    --generated_video_dir output_generated_video_dir \
    --device Ascend \
    --sample_fps 10 \
    --sample_rate 1 \
    --num_frames 65 \
    --height 480 \
    --width 640 \
    --num_workers 8 \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae \
    --enable_tiling \
    --tile_overlap_factor 0.125 \
    --save_memory \
    --ms_checkpoint /path/to/ms/checkpoint \
```

Runing this command will generate reconstructed videos under the given `output_generated_video_dir`. You can then evalute some common metrics (e.g., ssim, psnr) using the script under `opensora/eval/script`.


####  Performance

Here, we report the training performance and evaluation results on the UCF-101 dataset. Experiments are tested on Ascend 910* with mindspore 2.3.1 graph mode.

| model name  | cards  |  batch size | resolution | precision | discriminator |sink |recompute| jit level|  graph compile | s/step | img/s  | psnr | ssim  | recipe|
|:-----------|:------ |:-----------:|:----------:|:-------------:|:----------:|:------------:|:---:|:--------:|:--------:|--------:|------:|:----:|-------:|-------:|
| CausalVAE  |  8    |       1     | 25x256x256  |     BF16     |     FALSE    |  OFF |    OFF |    O0  |  3 mins   |     4.21   |  47.51 | 28.92 |    0.87    | [train](./scripts/causalvae/train_without_gan_loss_multi_device.sh) |
| CausalVAE  |  8    |      1      | 25x256x256  |  FP32     |     TRUE     |  OFF |    ON |    O0  |   3 mins    |    5.45 |   36.70  | 29.28 |   0.88   |  [train](./scripts/causalvae/train_with_gan_loss_multi_device.sh) |


### Training Diffusion Model

#### Preparation

**Step 1: Downloading Datasets**:


The [Open-Sora-Dataset-v1.2.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) contains annotation json files, which are listed below:

```text
Panda70M_HQ1M.json
Panda70M_HQ6M.json
sam_image_11185255_resolution.json
v1.1.0_HQ_part1.json
v1.1.0_HQ_part2.json
v1.1.0_HQ_part3.json
```

Please check the [readme doc](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) for details of these annotation files. [Open-Sora-Dataset-v1.2.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) contains the [Panda70M (training full)](https://drive.google.com/file/d/1DeODUcdJCEfnTjJywM-ObmrlVg-wsvwz/view?usp=sharing), [SAM](https://ai.meta.com/datasets/segment-anything/), and the data from [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main). You can take the following instructions only how to download [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main).


<details>
<summary> How to download Open-Sora-Dataset-v1.1.0? </summary>

The [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main) includes three image-text datasets and three video-text datasets. As reported in [Report v1.1.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.1.0.md), the three image-text datasets are:
| Name | Image Source | Text Captioner | Num pair |
|---|---|---|---|
| SAM-11M | [SAM](https://ai.meta.com/datasets/segment-anything/) |  [LLaVA](https://github.com/haotian-liu/LLaVA) |  11,185,255 |
| Anytext-3M-en | [Anytext](https://github.com/tyxsspa/AnyText) |  [InternVL-1.5](https://github.com/OpenGVLab/InternVL) |  1,886,137 |
| Human-160k | [Laion](https://laion.ai/blog/laion-5b/) |  [InternVL-1.5](https://github.com/OpenGVLab/InternVL) |  162,094 |


The three video-text datasets are:
| Name | Hours | Num frames | Num pair |
|---|---|---|---|
| [Mixkit](https://mixkit.co/) | 42.0h |  65 |  54,735 |
|   |  |  513 |  1,997 |
| [Pixabay](https://pixabay.com/) | 353.3h |  65 | 601,513 |
|   |  |  513 |  51,483 |
| [Pexel](https://www.pexels.com/) | 2561.9h |  65 |  3,832,666 |
|   |  |  513 |  271,782 |

Each video-text dataset has two annotation json files. For example, the mixkit dataset has `video_mixkit_65f_54735.json` which includes $54735$ video(65 frames)-text pairs and `video_mixkit_513f_1997.json` which includes $1997$ video(513 frames)-text pairs. The annotation json contains three keys: `path` corresponding to the video path, `cap` corresponding to the caption, and `frame_idx` corresponding to the frame indexes range. An example of annotation json file is shown below:

```json
[
  {
    "path": "Fish/mixkit-multicolored-coral-shot-with-fish-projections-4020.mp4",
    "frame_idx": "0:513",
    "cap": "The video presents a continuous exploration of a vibrant underwater coral environment,...",
  }
  ...
]
```


To prepare the training datasets, please first download the video and image datasets in [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main). We give a tutorial on how to download these datasets. See [downloading tutorial](./tools/download/README.md).

You need to download **at least one video dataset and one image dataset** to enable video-image joint training. After downloading all datasets, you can place images/videos under the folder `datasets`, which looks like:
```bash
datasets/
├───images/  # Human-160k
├───anytext3m/  # Anytext-3M-en
├───sam/  # SAM-11M
├───pixabay_v2/  # Pixabay
├───pexels/  # Pexel
└───mixkit/  # Mixkit
```
You can place the json files under the folder `anno_jsons`. The folder structure is:
```bash
anno_jsons/
├───video_pixabay_65f_601513.json
├───video_pixabay_513f_51483.json
├───video_pexel_65f_3832666.json
├───video_pexel_513f_271782.json
├───video_mixkit_65f_54735.json
├───video_mixkit_513f_1997.json
├───human_images_162094.json
├───anytext_en_1886137.json
└───sam_image_11185255.json
```
</details>


**Step 2: Extracting Embedding Cache**:

Next, please extract the text embeddings and save them in the disk for training acceleration. For each json file, you need to run the following command accordingly and save the t5 embeddings cache in the `output_path`.  

```bash
python opensora/sample/sample_text_embed.py \
    --data_file_path /path/to/caption.json \
    --output_path /path/to/text_embed_folder \
```

The text embeddings are extracted and saved under the specified `output_path`.

**Step 3: Revising the Paths**:

After extracting the embedding cache, you will have the following three paths ready:
```text
images/videos path: e.g., datasets/panda70m/
text embedding path: e.g., datasets/panda70m_emb-len=512/
annotation json path: e.g., datasets/anno_jsons/Panda70M_HQ1M.json
```
In the dataset file, for example, `scripts/train_data/merge_data.txt`, each line represents one dataset. Each line includes three paths: the images/videos folder, the text embedding cache folder, and the path to the annotation json file. Please revise them accordingly to the paths on your disk.


#### Example of Training Scripts

The training scripts are stored under `scripts/text_condition`. The single-device training scripts are under the `single-device` folder for demonstration. We recommend to use the parallel-training scripts under the `multi-devices` folder.

Here we choose an example of training scripts (`train_video3d_nx480p_zero2.sh`) and explain the meanings of some experimental arguments.

Here is the major command of the training script:
```shell
NUM_FRAME=29
python  opensora/train/train_t2v_diffusers.py \
    --data "scripts/train_data/merge_data.txt" \
    --num_frames ${NUM_FRAME} \
    --max_height 480 \
    --max_width 640 \
    --attention_mode xformers \
    --gradient_checkpointing \
    --pretrained "path/to/ms-or-safetensors-ckpt/from/last/stage" \
    --parallel_mode "zero" \
    --zero_stage 2 \
    # pass other arguments
```
There are some arguments related to the training dataset path:
- `data`: the text file to the video/image dataset. The text file should contain N lines corresponding to N datasets. Each line should have two or three items. If two items are available, they correspond to the video folder and the annotation json file. If three items are available, they correspond to the video folder, the text embedding cache folder, and the annotation json file.
- `num_frames`: the number of frames of each video sample.
- `max_height` and `max_width`: the frame maximum height and width.
- `attention_mode`: the attention mode, choosing from `math` or `xformers`. Note that we are not using the actual [xformers](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/xformers) library to accelerate training, but using MindSpore-native `FlashAttentionScore`. The `xformers` is kept for compatibility and maybe discarded in the future.
- `gradient_checkpointing`: it is referred to MindSpore [recomputation](https://www.mindspore.cn/docs/en/r2.3.1/api_python/mindspore/mindspore.recompute.html) feature, which can save memory by recomputing the intermediate activations in the backward pass.
- `pretrained`: the pretrained checkpoint to be loaded as initial weights before training. If not provided, the OpenSoraT2V will use random initialization. If provided, the path should be either the safetensors checkpoint directiory or path, e.g., "LanguageBind/Open-Sora-Plan-v1.2.0/1x480p" or "LanguageBind/Open-Sora-Plan-v1.2.0/1x480p/diffusion_pytorch_model.safetensors", or MindSpore checkpoint path, e.g., "t2i-image3d-1x480p/ckpt/OpenSoraT2V-ROPE-L-122.ckpt".
- `parallel_mode`: the parallelism mode chosen from ["data", "optim", "zero"], which denotes the data parallelism, the optimizer parallelism and the deepspeed zero_x parallelism.
- `zero_stage`: runs parallelism like deepspeed, supporting zero0, zero1, zero2, and zero3, if parallel_mode is "zero".

For the stage 4 (`29x720p`) and stage 5 (`93x720p`) training script, please refer to `train_video3d_29x720p_zero2_sp.sh` and `train_video3d_93x720p_zero2_sp.sh`.

#### Validation During Training

We also support to run validation during training. This is supported by editing the training script like this:
```diff
- --data "scripts/train_data/merge_data.txt" \
+ --data "scripts/train_data/merge_data_train.txt" \
+ --val_data "scripts/train_data/merge_data_val.txt" \
+ --validate True \
+ --val_batch_size 1 \
+ --val_interval 1 \
```
The edits allow to compute the loss on the validation set specified by `merge_data_val.txt` for every 1 epoch (defined by `val_interval`). `merge_data_val.txt` has the same format as `merge_data_train.txt`, but specifies a different subset from the train set. The validation loss will be recorded in the `result_val.log` under the output directory. For example training script, please refer to `train_video3d_29x720p_zero2_sp_val.sh` under `scripts/text_conditions/multi-devices/`.


#### Sequence Parallelism

We also support training with sequence parallelism and zero2 parallelism together. This is enabled by setting `--sp_size` and `--train_sp_batch_size`.  For example, with `sp_size=8` and `train_sp_batch_size=4`, 2 NPUs are used for a single video sample.

See `train_video3d_29x720p_zero2_sp.sh` under `scripts/text_condition/mult-devices/` for detailed usage.

#### Tips on Finetuning

To align with the hyper-parameters, we use the same learning rate (LR) $1e^{-4}$ as [Open-Sora-Plan v1.2.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/v1.2.0). However, our experience indicates that $1e^{-4}$ might be too large for finetuning the model on a small training set. If you want to finetune Open-Sora-Plan on your custom data with a small size, and notice that the large LR leads to unstable training, we have a few tips for you:

1. You can lower your LR or increase the effective batch size, for example, by increasing `gradient_accumulation_steps` or running multi-machine training.
2. You can try a different LR scheduler, for example, you can change the current constant LR scheduler to `polynomial decay` by:
```diff
-  --lr_scheduler="constant" \
+  --lr_scheduler="polynomial_decay" \
+  --lr_decay_steps=1000000 \
```
The edits will set the polynomial_decay LR scheduler, and decay the start LR to the end LR in 1000000 steps. You can adjust `lr_decay_steps` based on your `max_train_steps`. See other options of LR scheduler in `mindone/trainers/lr_schedule.py`.

#### Performance

The training performance are tested on ascend 910* with mindspore 2.3.1 graph mode. The results are as follows.

| model name      | cards       |  stage     |batch size   | num frames| resolution  | graph compile | parallelism | recompute | sink | jit level| s/step | img/s |
|:----------------|:----------- |:----------|:---------:|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|-------------------:|:----------:|
| OpenSoraT2V-ROPE-L-122 |  8   | 2 | 8  |           1 | 640x480     |  3mins     |         zero2                     | ON | ON | O0 |    2.35      | 27.23 |
| OpenSoraT2V-ROPE-L-122 |  8   | 3 |  1  |          29 | 640x480    |    6mins    |       zero2                      |  ON | ON | O0 |     3.68     | 63.04 |
| OpenSoraT2V-ROPE-L-122 |  8   | 4 |  1  |          29 |1280x720   |   10mins    |       zero2 + SP(sp_size=8)      |  OFF | ON | O0 |    4.32     | 6.71 |
| OpenSoraT2V-ROPE-L-122 |  8   | 5 |  1  |          93 | 1280x720   |   15mins    |       zero2 + SP(sp_size=8)      |  ON | ON | O0 |    24.40     | 3.81  |

> SP: sequence parallelism.
>
> Stage means the muti-stage training as illustrated above.

> batch size: the local batch size for a single card.

## 👍 Acknowledgement
* [Latte](https://github.com/Vchitect/Latte): The **main codebase** we built upon and it is an wonderful video generated model.
* [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT): Video Generation using VQ-VAE and Transformers.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [FiT](https://github.com/whlzy/FiT): Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595): Extending Context Window of Large Language Models via Positional Interpolation.
