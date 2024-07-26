# Open-Sora Plan

Here we provide an efficient MindSpore version of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main) from Peking University. We would like to express our gratitude to their contributions! :+1:

**OpenSora-PKU is still under active development.** Currently, we are in line with **Open-Sora-Plan v1.1.0**.

## 📰 News & States

|        Official News from OpenSora-PKU  | MindSpore Support     |
| ------------------ | ---------- |
| **[2024.05.27]** 🚀🚀🚀 PKU launched Open-Sora Plan v1.1.0, which significantly improves video quality and length, and is fully open source! Please check out their latest [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.1.0.md). | ✅ V.1.1.0 CausalVAE inference and LatteT2V infernece & three-stage training (`65x512x512`, `221x512x512`, `513x512x512`) |
| **[2024.04.09]** 🚀 PKU shared the latest exploration on metamorphic time-lapse video generation: [MagicTime](https://github.com/PKU-YuanGroup/MagicTime), and the dataset for train (updating): [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset).| N.A.  |
| **[2024.04.07]** 🔥🔥🔥 PKU released Open-Sora-Plan v1.0.0. See their [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md). | ✅ CausalVAE+LatteT2V+T5 inference and three-stage training (`17×256×256`, `65×256×256`, `65x512x512`)  |
| **[2024.03.27]** 🚀🚀🚀 PKU released the report of [VideoCausalVAE](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Train_And_Eval_CausalVideoVAE.md), which supports both images and videos.  | ✅ CausalVAE training and inference |
| **[2024.03.10]** 🚀🚀🚀 PKU supports training a latent size of 225×90×90 (t×h×w), which means to **train 1 minute of 1080P video with 30FPS** (2× interpolated frames and 2× super resolution) under class-condition.| frame interpolation and super-resolution are under-development.|
| **[2024.03.08]** PKU support the training code of text condition with 16 frames of 512x512. |   ✅ CausalVAE+LatteT2V+T5 training (`16x512x512`)|
| **[2024.03.07]** PKU support training with 128 frames (when sample rate = 3, which is about 13 seconds) of 256x256, or 64 frames (which is about 6 seconds) of 512x512. | class-conditioned training is under-development.|

[PKU Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) is under rapid development, and currently we have aligned our implementation with its code version on [20240611](https://github.com/PKU-YuanGroup/Open-Sora-Plan/commit/b08681f697658c81361e1ec6c07fba55c79bb4bd).  

## 🎥 Demo

The following videos are generated based on MindSpore and Ascend 910*.

<summary>Open-Sora-Plan v1.1.0 Demo</summary>


| 221×512×512 (9.2s) | 221×512×512 (9.2s) | 221×512×512 (9.2s) |
| --- | --- | --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f221/An%20aerial%20shot%20of%20a%20lighthouse%20standing%20tall%20on%20a%20rocky%20cliff,%20its%20beacon%20cutting%20through%20the%20early%20.gif?raw=true" width=224> | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f221/The%20camera%20rotates%20around%20a%20large%20stack%20of%20vintage%20televisions%20all%20showing%20different%20programs-1950s.gif?raw=true" width=224>  | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f221/The%20video%20captures%20the%20spectacle%20of%20a%20continuous%20fireworks%20show%20against%20the%20backdrop%20of%20a%20starry%20nig.gif?raw=true" width=224> |
| An aerial shot of a lighthouse standing tall on a rocky cliff... | The camera rotates around a large stack of vintage televisions...  | The video captures the spectacle of a continuous fireworks...  |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f221/Aerial%20view%20of%20Santorini%20during%20the%20blue%20hour%2C%20showcasing%20the%20stunning%20architecture%20of%20white%20Cycladi.gif?raw=true" width=224> |<img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f221/Drone%20shot%20along%20the%20Hawaii%20jungle%20coastline%2C%20sunny%20day.%20Kayaks%20in%20the%20water.gif?raw=true" width=224> | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f221/The%20video%20presents%20an%20abstract%20composition%20centered%20around%20a%20hexagonal%20shape%20adorned%20with%20a%20starburs.gif?raw=true" width=224>  |
| Aerial view of Santorini during the blue hour... | Drone shot along the Hawaii jungle coastline...  | The video presents an abstract composition centered around a hexagonal shape...  |


| 65×512×512 (2.7s) | 65×512×512 (2.7s) | 65×512×512 (2.7s) |
| --- | --- | --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f65/0-3D%20animation%20of%20a%20small,%20round,%20fluffy%20creature%20with%20big,%20expressive%20eyes%20explores%20a%20vibrant,%20enchan.gif?raw=true" width=224> | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f65/0-A%20corgi%20vlogging%20itself%20in%20tropical%20Maui..gif?raw=true" width=224>  | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f65/0-A%20painting%20of%20a%20boat%20on%20water%20comes%20to%20life,%20with%20waves%20crashing%20and%20the%20boat%20becoming%20submerged..gif?raw=true" width=224> |
| 3D animation of a small, round, fluffy creature with... | A corgi vlogging itself in tropical Maui.  | A painting of a boat on water comes to life...  |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f65/0-A%20solitary%20astronaut%20plants%20a%20flag%20on%20an%20alien%20planet%20covered%20in%20crystal%20formations.%20The%20shot%20tracks.gif?raw=true" width=224> |<img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f65/0-Extreme%20close-up%20of%20chicken%20and%20green%20pepper%20kebabs%20grilling%20on%20a%20barbeque%20with%20flames.%20Shallow%20focu.gif?raw=true" width=224> | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/f65/0-In%20an%20ornate,%20historical%20hall,%20a%20massive%20tidal%20wave%20peaks%20and%20begins%20to%20crash.%20Two%20surfers,%20surfing..gif?raw=true" width=224>  |
| A solitary astronaut plants a flag on an alien planet... | Extreme close-up of chicken and green pepper kebabs...  | In an ornate, historical hall, a massive tidal wave...  |



Videos are saved to `.gif` for display. See the text prompts in `examples/prompt_list_65.txt` and `examples/prompt_list_221.txt`.

## 🔆 Features

- 📍 **Open-Sora-Plan v1.1.0** with the following features
    - ✅ Sequence parallelism
    - ✅ CausalVAE-4x8x8 training and inference. Supports video reconstruction.
    - ✅ T5 TextEncoder model inference.
    - ✅ Text-to-video generation in 512x512 resolution and up to 513 frames.
    - ✅ Three-stage training: i) 65x512x512 pretraining; ii) 221x512x512 finetuning; iii) 513x512x512 finetuning.
    - ✅ Acceleration methods: flash attention, recompute (graident checkpointing), mixed precision, data parallelism, optimizer-parallel, etc..
    - ✅ Evaluation metrics : PSNR and SSIM.


### TODO
* [ ] Scaling model parameters and dataset size **[WIP]**.
* [ ] Evaluation of various metrics **[WIP]**.

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
* [Training](#training)
* [Acknowledgement](#acknowledgement)

Other useful documents and links are listed below.

* Repo structure: [structure.md](docs/structure.md)


## Installation
1. Use python>=3.8 [[install]](https://www.python.org/downloads/)

2. Install MindSpore 2.3 master (0705daily) according to the [website](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/). Select the corresponding wheel file based your computer's OS and the python verison. Please use C18 CANN (0705) which can be downloaded from [here](https://repo.mindspore.cn/ascend/ascend910/20240705/).


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

### Open-Sora-Plan v1.1.0 Model Weights

Please download the torch checkpoint of T5 from [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl), and download the opensora v1.1.0 models' weights from [LanguageBind/Open-Sora-Plan-v1.1.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main). Place them under `examples/opensora_pku` as shown below:
```bash
opensora_pku
├───LanguageBind
│   └───Open-Sora-Plan-v1.1.0
│       ├───221x512x512
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
    --model_path LanguageBind/Open-Sora-Plan-v1.1.0/vae \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --device Ascend \
    --sample_rate 1 \
    --num_frames 513 \
    --resolution 256 \
    --crop_size 256 \
    --ae CausalVAEModel_4x8x8
```
Please change the `--video_path` to the existing video file path and `--rec_path` to the reconstructed video file path. You can set `--grid` to save the original video and the reconstructed video in the same output file.

You can also run video reconstruction given an input video folder. See `scripts/causalvae/gen_video.sh`.

### Open-Sora-Plan v1.1.0 Command Line Inference

You can run text-to-video inference on a single Ascend device using the script `scripts/text_condition/sample_video_65.sh` or `scripts/text_condition/sample_video_221.sh`.
```bash
python opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.1.0 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_65.txt \
    --ae CausalVAEModel_4x8x8 \
    --version 65x512x512 \
    --num_frames 65 \
    --height 512 \
    --width 512 \
    --save_img_path "./sample_videos/prompt_list_65" \
    --fps 24 \
    --guidance_scale 7.5 \
    --num_sampling_steps 150 \
    --enable_tiling
```
You can change the `version` to `221x512x512` to change the number of frames and resolutions.

> In case of OOM error, there are two options:
> 1. Pass `--enable_time_chunk True` to allow vae decoding temporal frames as small, overlapped chunks. This can reduce the memory usage, which sacrificies a bit of temporal consistency.
> 2. Seperate the inference into two stages. In stage 1, please run inference with `--save_latents`. This will save some `.npy` files in the output directory. Then in stage 2, please run the same inference script with `--decode_latents`. The generated videos will be saved in the output directory.

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


## Training

### Causal Video VAE

#### Preparation

**Step 1: Downloading Datasets**:

To train the causal vae model, you need to prepare a video dataset. You can download this video dataset from [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main). We give a tutorial on how to download these datasets. See [downloading tutorial](./tools/download/README.md).

**Step 2: Converting Pretrained Weights**:

Causal video vae can be initialized from vae 2d for better convergence. This can be done by inflating the 2d vae model checkpoint as follows:

```
python tools/model_conversion/inflate_vae2d_to_vae3d.py \
    --src /path/to/sd-vae-ft-mse.ckpt  \
    --target pretrained/causal_vae_488_init.ckpt
```
> In case you lack vae 2d checkpoint in mindspore format, please use `tools/model_conversion/convert_vae_2d.py` for model conversion. After downloading the [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) weights, you can run `python tools/model_conversion/convert_vae_2d.py --src path/to/diffusion.safetensor --target /path/to/sd-vae-ft-mse.ckpt`.

Please also download [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) and put it under `pretrained/` for training with lpips loss.

#### Standalone Training

After downloading the datasets and preparing the pretrained weight, you can revise the `--video_path` in the training script to the video folder path of your downloaded dataset. This will allow the training script to load all video files under `video_path` in a **recursive manner**, and use them as the training data. Make sure the `--load_from_checkpoint` is set to the pretrained weight, e.g., `pretrained/causal_vae_488_init.ckpt`.


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

To launch a single-card training using perceputal loss, you can refer to `scripts/causalvae/train_without_gan_loss.sh` and run:
```bash
bash scripts/causalvae/train_without_gan_loss.sh
```

If you want to train causalvae with perceputal loss and GAN loss, you can refer to `scripts/causalvae/train_with_gan_loss.sh`.

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

#### Multi-Stage Training

As stated in [Training Details](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.1.0.md#training-details), the authors trained for 100k steps in the first stage with a video shape of 9×256×256. Then they increased the frame count from 9 to 25 and found that this significantly improved the model's performance. In the first two stages, they enabled the learnable mixed factor in `TimeUpsampleRes2x` and `TimeDownsampleRes2x`. In the third stage, they reinitialized the mixed factor to 0.5 (sigmoid(0.5) = 0.6225) to further enhance the model's capabilities.

You can revise `--video_num_frames` and `--resolution` in the training scripts under `scripts/causalvae/` for each stage accordingly.



#### Inference After Training

After training, you will find the checkpoint files under the `ckpt/` folder of the output directory. To evaluate the reconstruction of the checkpoint file, you can take `scripts/causalvae/gen_video.sh` and revise it like:

```bash
python examples/rec_video_vae.py \
    --batch_size 1 \
    --real_video_dir input_real_video_dir \
    --generated_video_dir output_generated_video_dir \
    --device Ascend \
    --sample_fps 10 \
    --sample_rate 1 \
    --num_frames 9 \  # revise according to your training stage
    --resolution 256 \ # revise according to your training stage
    --crop_size 256 \ # revise according to your training stage
    --num_workers 8 \
    --ckpt /path/to/your/.ckpt/file
```

Runing this command will generate reconstructed videos under the given `output_generated_video_dir`. You can then evalute some common metrics (e.g., ssim, psnr) using the script under `opensora/eval/script`.

#### Performance

Taking the stage-1 training as an example, we record the training speed as follows:

| Model           | Context        | Precision | BS  | NPUs | num_frames | Resolution  | With GAN loss  | Train T. (s/step) |
|:----------------|:---------------|:----------|:---:|:----:|:-----------------------:|:-----------:|:-----------:|:-----------------:|
| CausalVAE_4x8x8  | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  1  |  8   |         9         | 256x256     |  False |     0.97      |
| CausalVAE_4x8x8  | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | FP32      |  1  |  8   |         9         | 256x256     |  True |     1.63        |

#### Example of Training Experiment

To validate the training script, we run 8-card parallel training of CausalVAE_4x8x8 with GAN loss using the [UCF-101 dataset](https://www.crcv.ucf.edu/research/data-sets/ucf101/). The training set consists of 10656 videos and the test set consists of 2664 videos.

We revise the `video_path` of `scripts/causalvae/train_with_gan_loss_multi_device.sh` to the UCF-101 training set, and then start training. After training, we run inference with the checkpoint using `scripts/causalvae/gen_video.sh` to save the generated videos. Then we revise the `real_video_dir` and `generated_video_dir` in `opensora/eval/scripts/cal_ssim.sh` to the video folder of the test set and the video folder of the generated videos to evaluate SSIM scores. The similar process is needed to evaluate PSNR scores using `opensora/eval/scripts/cal_psnr.sh`

Here are the evaluation metrics of the checkpoint file trained for 135k steps:

| Train Steps | With GAN loss | PSNR | SSIM |
| --- | ---| ---|---|
|135000 | True|29.8343 | 0.8893|


### Training Diffusion Model

#### Preparation

**Step 1: Downloading Datasets**:

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

**Step 2: Extracting Embedding Cache**:

Next, please extract the text embeddings and save them in the disk for training acceleration. For each json file, you need to run the following command accordingly and save the t5 embeddings cache in the `output_path`.  

```bash
python opensora/sample/sample_text_embed.py \
    --data_file_path /path/to/caption.json \
    --output_path /path/to/text_embed_folder \
```
To extract text embeddings for all annotation json files using a single card, you can refer to `scripts/embedding_cache/extract_all.sh`. If you want to try extracting embedding cache using multiple cards in a single node, please refer to `scripts/embedding_cache/extract_multi.sh`.

The text embeddings are extracted and saved under the specified `output_path`. The `output_path` folder structure is similar to, e.g.,:
```bash
datasets/
├───mixkit-t5-emb-len=300_65f/
│   ├───Airplane/
│   │       ├───[video-file-name]-frame_idx-0:65.npz
│   │       ├───[video-file-name]-frame_idx-65:130.npz
│   │       └───...
│   └───...
└───mixkit-t5-emb-len=300_513f/
    ├───Airplane/
    │       ├───[video-file-name]-frame_idx-0:513.npz
    │       ├───[video-file-name]-frame_idx-513:1026.npz
    │       └───...
    └───...
```
**Step 3: Revising the Paths**:

After extracting the embedding cache, you will have the following three paths ready:
```text
images/videos path: e.g., datasets/images/
t5 embedding path: e.g., datasets/images-t5-emb-len=300/
annotation json path: e.g., anno_jsons/human_images162094.json
```
In the dataset file, for example, `scripts/train_data/image_data.txt`, each line represents one dataset. Each line includes three paths: the images/videos folder, the t5 embedding cache folder, and the path to the annotation json file. Please revise them accordingly to the paths on your disk.

**Step 4: Converting Pretrained Weights**:

The first-stage training depends on the `t2v.pt` from [Vchitect/Latte](https://huggingface.co/maxin-cn/Latte/tree/main). Please download `t2v.pt` and place it under `LanguageBind/Open-Sora-Plan-v1.1.0/t2v.pt`. Then run model conversion with:
```bash
python tools/model_conversion/convert_latte.py \
  --src LanguageBind/Open-Sora-Plan-v1.1.0/t2v.pt \
  --target LanguageBind/Open-Sora-Plan-v1.1.0/t2v.ckpt
```

> **Since [Vchitect/Latte](https://huggingface.co/maxin-cn/Latte/tree/main) has deleted `t2v.pt` from their HF repo, please download `t2v.ckpt` from this [URL](https://download-mindspore.osinfra.cn/toolkits/mindone/opensora-pku/t2v.ckpt). There is no need to convert it.**

#### Example of Training Scripts
Here we choose an example of training scripts (`train_videoae_65x512x512.sh`) and explain the meanings of some experimental arguments. This is an example of parallel training script which uses data parallelism. If you want to try single-device training, please refer to `train_videoae_65x512x512_single_device.sh`.

There some hyper-parameters that may vary between different experiments:
```shell
image_size=512  # the image size of frames, same to image height and image width
use_image_num=4  # to include n number of images in an input sample
num_frames=65  # to sample m frames from a single video. The total number of images： num_frames + use_image_num
model_dtype="bf16" # the data type used for mixed precision of the diffusion transformer model (LatteT2V).
amp_level="O2" # the default auto mixed precision level for LatteT2V.
enable_flash_attention="True" # whether to use MindSpore Flash Attention
batch_size=2 # training batch size
lr="2e-05" # learning rate. Default learning schedule is constant
```

Here is the major command of the parallel-training script:
```shell
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9000 --log_dir=$output_dir/parallel_logs opensora/train/train_t2v.py \
      --video_data "scripts/train_data/video_data.txt" \
      --image_data "scripts/train_data/image_data.txt" \
      --pretrained LanguageBind/Open-Sora-Plan-v1.1.0/t2v.ckpt \
    ... # pass other arguments
```
We use `msrun` to launch the parallel training tasks. For single-node multi-device training, `worker_num` and `local_worker_num` should be the same to the number of training devices.  `master_port` specifies the scheduler binding port number.

There are some arguments related to the training dataset path:
- `video_data` or `image_data`: the text file to the video/image dataset. The text file should contain N lines corresponding to N datasets. Each line should have two or three items. If two items are available, they correspond to the video folder and the annotation json file. If three items are available, they correspond to the video folder, the text embedding cache folder, and the annotation json file.
- `pretrained`: the pretrained checkpoint to be loaded as initial weights before training. If not provided, the LatteT2V will use random initialization.

For the detailed explanations for other arguments, please refer to the document for [training arguments](docs/training_args.md).

> Note:
> - In Graph mode (default), MindSpore takes about 10~20 mins for graph compilation.
> - For acceleration, we set the `dataset_sink_mode` to True by default. For more information about data sink mode, see [MindSpore doc for data sink](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.data_sink.html).
> - Supports resume training by setting `--resume_training_checkpoint True`.

#### Parallel Training

Before launching the first-stage training, please make sure you set the three paths correctly in `image_data.txt` and `video_data.txt`.

```bash
# start 65x512x512 pretraining, 8 NPUs
bash scripts/text_condition/train_videoae_65x512x512.sh
```
During training, the training logs will be saved under `parallel_logs/` folder of the specified output directory, e.g., `parallel_logs/worker_0.log`. The loss values and average per step time will saved in `result.log` in the output directory.

After the first-stage training, if data parallelism (the default parallel mode) is applied, the checkpoint files will be saved under `ckpt/` folder. If optimizer parallelism is applied (setting `--parallel_mode` to "optim"), there will be multiple checkpoint shards saved in the `ckpt/`. See the following method on how to merge multiple checkpoint shards into a full checkpoint file.
<details>
<summary>How to merge multiple checkpoint shards</summary>

Please run the following command to combine the multiple checkpoint shards into a full one:
```
python tools/ckpt/combine_ckpt.py --src output_dir/ckpt --dest output_dir/ckpt --strategy_ckpt output_dir/src_strategy.ckpt
```
Afterwards, you will obtain a full checkpoint file under `output_dir/ckpt/rank_0/full_0.ckpt`.

</details>

> Note:
> - If you want to run inference with a checkpoint file, please revise the script `scripts/text_condition/sample_video.sh` and append `--pretrained_ckpt path/to/your.ckpt` to the end of the inference command.

Then please revise `scripts/text_condition/train_videoae_221x512x512.sh`, and change `--pretrained` to the full checkpoint path from the `65x512x512` stage. Then run:

```bash
# start 221x512x512 finetuning, 8 NPUs
bash scripts/text_condition/train_videoae_221x512x512_sp.sh
```
> Note:
> - You can try modifying `--dataloader_num_workers` and `--dataloader_prefetch_size` on `train_videoae_221x512x512_sp.sh` to speed up when you have enough cpu memory.

Simiarly, please revise the `--pretrained` to the checkpoint path from the `221x512x512` stage, and then start the third-stage training:

```bash
# start 513x512x512 finetuning, 8 NPUs
bash scripts/text_condition/train_videoae_513x512x512_sp.sh
```
> Note:
> - You can try modifying `--dataloader_num_workers` and `--dataloader_prefetch_size` on `train_videoae_513x512x512_sp.sh` to speed up when you have enough cpu memory.


#### Overfitting Experiment

To verify the training script and convergence speed, we performed an overfitting experiment: training the stage 1 model $(65+4)\times512\times512$ on 64 videos selected from the mixkit dataset. The stage 1 model was intialized with `t2v.ckpt`, and we trained it with the hyper-parameters listed in `scripts/text_condition/train_videoae_65x512x512.sh`, except that we only trained it on 64 videos for 3000 steps.

The checkpoint after 3000 steps generated videos similar to the original videos, which means the convergence of the overfitting experiment was as good as we expected. Some generated videos are shown below:

| 65×512×512 (2.7s) | 65×512×512 (2.7s) | 65×512×512 (2.7s) |
| --- | --- | --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/overfit-fp65/0-a%20lively%20scene%20at%20a%20ski%20resort%20nestled%20in%20the%20heart%20of%20a%20snowy%20mountain%20range.%20From%20a%20high%20vantage%20p.gif?raw=true" width=224> | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/overfit-fp65/0-a%20serene%20scene%20of%20a%20clear%20blue%20sky.%20Dominating%20the%20top%20right%20corner%20of%20the%20frame%20is%20a%20single,%20fluffy.gif?raw=true" width=224>  | <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1.1/t2v/overfit-fp65/0-an%20aerial%20view%20of%20a%20rugged%20landscape.%20Dominating%20the%20scene%20are%20large,%20jagged%20rocks%20that%20cut%20across%20e.gif?raw=true" width=224> |
| a lively scene at a ski resort... | a serene scene of a clear blue sky...  | an aerial view of a rugged landscape...  |


#### Performance

We evaluated the training performance on MindSpore and Ascend NPUs. The results are as follows.

| Model           | Context        | Precision | BS  | NPUs | num_frames + num_images | Resolution  | Train T. (s/step) |
|:----------------|:---------------|:----------|:---:|:----:|:-----------------------:|:-----------:|:-----------------:|
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  2  |  8   |         17 + 4          | 512x512     |       2.45        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  2  |  8   |         65 + 16         | 512x512     |       9.36       |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  2  |  8   |         65 + 4          | 512x512     |       7.02        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  1  |  8   |         221 + 4         | 512x512     |       7.18        |
| LatteT2V-XL/122 | D910\*-[CANN C18(0705)](https://repo.mindspore.cn/ascend/ascend910/20240705/)-[MS2.3_master(0705)](https://repo.mindspore.cn/mindspore/mindspore/version/202407/20240705/master_20240705220018_51f414917fd9a312dd43ea62eea61cf37c3dfbd6_newest/unified/) | BF16      |  1  |  8   |         513 + 8         | 512x512     |        12.3       |

> Context: {NPU type}-{CANN version}-{MindSpore version}

See [Performance Boosting History](docs/performance_boosting_history.md) 🚀🚀🚀 on how we achieve the state of the art performance.

## 👍 Acknowledgement
* [Latte](https://github.com/Vchitect/Latte): The **main codebase** we built upon and it is an wonderful video generated model.
* [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT): Video Generation using VQ-VAE and Transformers.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [FiT](https://github.com/whlzy/FiT): Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595): Extending Context Window of Large Language Models via Positional Interpolation.
