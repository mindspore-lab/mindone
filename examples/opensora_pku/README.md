# Open-Sora Plan

Here we provide an efficient MindSpore version of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main) from Peking University. We would like to express our gratitude to their contributions! :+1:

**OpenSora-PKU is still under active development.** Currently, we are in line with **Open-Sora-Plan v1.3.0** ([commit id](https://github.com/PKU-YuanGroup/Open-Sora-Plan/commit/9fa322fbbb276e2bbe40b2f439e3d610af3d7690)).

## 📰 News & States

|        Official News from OpenSora-PKU  | MindSpore Support     |
| ------------------ | ---------- |
| **[2024.10.16]** 🎉 PKU released version 1.3.0, featuring: **WFVAE**, **pompt refiner**, **data filtering strategy**, **sparse attention**, and **bucket training strategy**. They also support 93x480p within **24G VRAM**. More details can be found at their latest [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.3.0.md). | ✅ V.1.3.0 WFVAE and OpenSoraT2V: inference, multi-stage & multi-devices training |
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

<summary>Open-Sora-Plan v1.3.0 Demo</summary>

93x352x640 Text-to-Video Generation.

| 93x352x640 (5.8s) |
| --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1/v1.3/0-A-litter-of-golden-retriever-puppies-playing-in-the-snow.Their-heads-pop-out-of-the-snow--covered-in-0.gif?raw=true" width=640> |
| A litter of golden retriever puppies playing in the snow... |

| 93x352x640 (5.8s) |
| --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1/v1.3/0-An-extreme-close-up-of-an-gray-haired-man-with-a-beard-in-his-60s--he-is-deep-in-thought-pondering-t-0.gif?raw=true" width=640>  |
| An extreme close-up of an gray-haired man with a beard in his 60s...  |

| 93x352x640 (5.8s) |
| --- |
| <img src="https://github.com/wtomin/mindone-assets/blob/main/opensora_pku/v1/v1.3/0-Drone-view-of-waves-crashing-against-the-rugged-cliffs-along-Big-Sur-s-garay-point-beach.The-crashin-0.gif?raw=true" width=640> |
| Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach...  |

Videos are saved to `.gif` for display.


## 🔆 Features

- 📍 **Open-Sora-Plan v1.3.0** with the following features
    - ✅ WFVAE inference & multi-stage training.
    - ✅ mT5-xxl TextEncoder model inference.
    - ✅ Prompt Refiner Inference.
    - ✅ Text-to-video generation up to 93 frames and 640x640 resolution.
    - ✅ Multi-stage training using Zero2 and sequence parallelism.
    - ✅ Acceleration methods: flash attention, recompute (graident checkpointing), mixed precision, data parallelism, etc..
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

### Open-Sora-Plan v1.3.0 Model Weights

Please download the torch checkpoint of mT5-xxl from [google/mt5-xxl](https://huggingface.co/google/mt5-xxl/tree/main), and download the opensora v1.3.0 models' weights from [LanguageBind/Open-Sora-Plan-v1.3.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main). Place them under `examples/opensora_pku` as shown below:
```bash
mindone/examples/opensora_pku
├───LanguageBind
│   └───Open-Sora-Plan-v1.3.0
│       ├───any93x640x640/
│       ├───any93x640x640_i2v/
│       ├───prompt_refiner/
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
python tools/model_conversion/convert_wfvae.py --src LanguageBind/Open-Sora-Plan-v1.3.0/vae/merged.ckpt --target LanguageBind/Open-Sora-Plan-v1.3.0/vae/diffusion_pytorch_model.safetensors  --config LanguageBind/Open-Sora-Plan-v1.3.0/vae/config.json

python tools/model_conversion/convert_pytorch_ckpt_to_safetensors.py --src google/mt5-xxl/pytorch_model.bin --target google/mt5-xxl/model.safetensors  --config google/mt5-xxl/config.json
```

In addition, please merge the multiple .saftensors files under `any93x640x640/` into a merged checkpoint:
```shell
python tools/ckpt/merge_safetensors.py -i LanguageBind/Open-Sora-Plan-v1.3.0/any93x640x640/ -o LanguageBind/Open-Sora-Plan-v1.3.0/diffusion_pytorch_model.safetensors  -f LanguageBind/Open-Sora-Plan-v1.3.0/any93x640x640/diffusion_pytorch_model.safetensors.index.json
```

Once the checkpoint files have all been prepared, you can refer to the inference guidance below.

## Inference

### CausalVAE Command Line Inference

You can run video-to-video reconstruction task using `scripts/causalvae/single-device/rec_video.sh`:
```bash
python examples/rec_video.py \
    --ae "WFVAEModel_D8_4x8x8" \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --device Ascend \
    --sample_rate 1 \
    --num_frames 65 \
    --height 512 \
    --width 512 \
    --fps 30 \
    --enable_tiling
```
Please change the `--video_path` to the existing video file path and `--rec_path` to the reconstructed video file path. You can set `--grid` to save the original video and the reconstructed video in the same output file.

You can also run video reconstruction given an input video folder. See `scripts/causalvae/single-device/rec_video_folder.sh`.

### Open-Sora-Plan v1.3.0 Command Line Inference

You can run text-to-video inference on a single Ascend device using the script `scripts/text_condition/single-device/sample_t2v_93x640.sh`.
```bash
# Single NPU
python opensora/sample/sample.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.3.0/any93x640x640 \
    --version v1_3 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --text_encoder_name_1 google/mt5-xxl \
    --text_prompt examples/sora.txt \
    --ae WFVAEModel_D8_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --save_img_path "./sample_videos/sora_93x640_mt5" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --seed 1234 \
    --num_samples_per_prompt 1 \
    --rescale_betas_zero_snr \
    --prediction_type "v_prediction" \
    --precision bf16 \
```
You can change the `num_frames`, `height` and `width`. Note that DiT model is trained arbitrarily on stride=32.
So keep the resolution of the inference a multiple of 32. `num_frames` needs to be 4n+1, e.g. 93, 77, 61, 45, 29, 1.


If you want to run a multi-device inference using data parallelism, please use `scripts/text_condition/multi-devices/sample_t2v_93x640_ddp.sh`.
The script will run a 8-card inference and save the log files into "parallel_logs/". `--master_port` specifies the scheduler binding port number. `--worker_num` and `--local_worker_num` should be the same to the number of running devices, e.g., 8.

In case of the following error:
```bash
RuntimtError: Failed to register the compute graph node: 0. Reason: Repeated registration node: 0
```

Please edit the `master_port` to a different port number in the range 1024 to 65535, and run the script again.

See more examples of multi-device inference scripts under `scripts/text_condifion/multi-devices`.


### Prompt Refiner Inference

If you want to run T2V inference with caption refiner, you should attach following argument to the T2V inference command above:
```
  --caption_refiner "LanguageBind/Open-Sora-Plan-v1.3.0/prompt_refiner/"
```

If you just want to run prompt refinement, please run:
```bash
python opensora/sample/caption_refiner.py
```
### Sequence Parallelism

We support running inference with sequence parallelism. Please see the `sample_t2v_93x640_sp.sh` under `scripts/text_condition/multi-devices/`. The script will run a 8-card inference with `sp_size=8`, which means each video tensor is sliced into 8 parts along the sequence dimension. If you want to try `sp_size=4`, you can revise it as below:
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
msrun --bind_core=True --worker_num=4 --local_worker_num=4 --master_port=9000 --log_dir="./sample_videos/sora_93x640_mt5_sp/parallel_logs/" \
   opensora/sample/sample.py \
    ... \
    --sp_size 4
```

## Training

### WFVAE

#### Preparation

**Step 1: Downloading Datasets**:

To train the causal vae model, you need to prepare a video dataset. Please download K400 dataset from [this repository](https://github.com/cvdfoundation/kinetics-dataset) as used in the [Arxiv paper](https://arxiv.org/abs/2411.17459) or download the UCF101 dataset from [the official website](https://www.crcv.ucf.edu/data/UCF101.php) as used in this tutorial.


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

**Step 2: Prepare Pretrained Weights**:

Open-Sora-Plan-v1.3.0 trains WFVAE in multiple stages. The loss used for the first two stages is a weighted sum of multiple loss terms:

$L = L_{recon} + \lambda_{adv}L_{adv} + \lambda_{KL}L_{KL} + \lambda_{WL}L_{WL}$

$L_{recon}$ represents the reconstruction loss (L1). $L_{adv}$ is the adversarial loss, and its weight $\lambda_{adv}$ is given by the argument `--disc_weight`. $L_{KL}$ is the KL divergence loss, and its weight $\lambda_{KL}$ is given by `--kl_weight`. $L_{WL}$ is the wavelet loss, and its weight $\lambda_{WL}$ is given by `--wavelet_weight`. In the third stage, LPIPS loss is also used to improve the performance. Its weight $\lambda_{lpips}$ is given by the argument `--perceptual_weight `. Please see more arguments in `opensora/train/train_causalvae.py`.

In order to train vae with LPIPS loss, please also download [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) and put it under `pretrained/`.

**Steps 3: Hyper-parameters Setting**

As introduced in the [Open-Sora Plan Arxiv paper](https://arxiv.org/abs/2412.00131), the hyper-parameters of each stage is summerized in the following table:
| Stage |  Resolution | Num of frames | FPS | Batch size  | Train Steps | Discrminator |  $\lambda_{lpips}$ |
|:---   |:---         |:---           |:--- |:---         |:---         |:---          |:---                |
| 1     | 256x256     | 25            | Original fps       |   8 |   800K     | TRUE         | -                  |
| 2     | 256x256     | 49            | Original fps / 2   |   8 |   200K     | TRUE         | -                  |
| 3     | 256x256     | 49            | Original fps  / 2  |   8 |   200K     | TRUE         | 0.1                |

See the hyper-parameters in `scripts/causalvae/multi-devices/train_stage_x.sh`

> Note:
> - We support resume training by setting `--resume_from_checkpoint True`. It is the same for the multi-device training script.
> - We also provide the standalone training script: `scripts/causalvae/single-device/train.sh`.

#### Inference After Training

After training, you will find the checkpoint files under the `ckpt/` folder of the output directory. To evaluate the reconstruction of the checkpoint file, you can take `scripts/causalvae/single-device/rec_video_folder.sh` and revise it like:

```bash
python examples/rec_video_folder.py \
    --batch_size 1 \
    --real_video_dir datasets/UCF-101/ \
    --data_file_path datasets/ucf101_test.csv \
    --generated_video_dir recons/ucf101_test/ \
    --device Ascend \
    --sample_fps 30 \
    --sample_rate 1 \
    --num_frames 25 \
    --height 256 \
    --width 256 \
    --num_workers 8 \
    --ae "WFVAEModel_D8_4x8x8" \
    --ae_path LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    --ms_checkpoint path/to/ms/ckpt \
```

Runing this command will generate reconstructed videos under the given `output_generated_video_dir`. You can then evalute some common metrics (e.g., ssim, psnr) using the script under `opensora/eval/script`.

### Training Diffusion Model

#### Preparation

**Step 1: Downloading Datasets**:

Open-Sora-Dataset-v1.3.0 dataset is the same as the dataset used in [Open-Sora-Dataset-v1.2.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) which contains annotation json files listed below:

```text
Panda70M_HQ1M.json
Panda70M_HQ6M.json
sam_image_11185255_resolution.json
v1.1.0_HQ_part1.json
v1.1.0_HQ_part2.json
v1.1.0_HQ_part3.json
```

Please check the [readme doc](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) for details of these annotation files. [Open-Sora-Dataset-v1.2.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0) contains the [Panda70M (training full)](https://drive.google.com/file/d/1DeODUcdJCEfnTjJywM-ObmrlVg-wsvwz/view?usp=sharing), [SAM](https://ai.meta.com/datasets/segment-anything/) and the data from [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main). You can take the following instructions on how to download [Open-Sora-Dataset-v1.1.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main).


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

Next, please extract the text embeddings and save them in the disk for training acceleration. For each json file, you need to run the following command accordingly and save the mt5-xxl embeddings cache in the `output_path`.  

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

**Step 4: Hyper-Parameters Setting**


As introduced in the [Open-Sora Plan Arxiv paper](https://arxiv.org/abs/2412.00131), the hyper-parameters of each stage is summerized in the following table:

| Stage |  Resolution | Num of frames | Datasets | Batch size  | Train Steps | LR |  Attention |
|:---   |:---         |:---           |:--- |:---         |:---         |:---          |:---                |
| 1 (T2I)    | 256x256     | 1             | SAM, AnyText, Human Images      |  1024 |   150K (full-attention) + 100K (skiparse attention)     | 2e-5         |  Full 3D -> Skiparse             |
| 2  (T2I&T2V)   |  maximumly 93×640×640     | 93            | SAM, Panda70M                   |  1024 |   200K     | 2e-5         | Skiparse           |
| 3  (T2V)   | 93x352x640   | 93            | filtered Panda70M, high-quality data  | 1024 |   100K~200K     | 1e-5         | Skiparse            |


#### Example of Training Scripts

The training scripts are stored under `scripts/text_condition`. The single-device training scripts are under the `single-device` folder for demonstration. We recommend to use the parallel-training scripts under the `multi-devices` folder.

Here we choose an example of training scripts (`train_t2i_stage1.sh`) and explain the meanings of some experimental arguments.

Here is the major command of the training script:
```shell
NUM_FRAME=1
WIDTH=256
HEIGHT=256
python opensora/train/train_t2v_diffusers.py \
    --data "scripts/train_data/image_data_v1_2.txt" \
    --num_frames ${NUM_FRAME} \
    --force_resolution \
    --max_height ${HEIGHT} \
    --max_width ${WIDTH} \
    --gradient_checkpointing \
    --pretrained path/to/last/stage/ckpt \
    --parallel_mode "zero" \
    --zero_stage 2 \
    # pass other arguments
```
There are some arguments related to the training dataset path:
- `data`: the text file to the video/image dataset. The text file should contain N lines corresponding to N datasets. Each line should have two or three items. If two items are available, they correspond to the video folder and the annotation json file. If three items are available, they correspond to the video folder, the text embedding cache folder, and the annotation json file.
- `num_frames`: the number of frames of each video sample.
- `max_height` and `max_width`: the frame maximum height and width.
- `force_resolution`: whether to train with fixed resolution or dynamic resolution. If `force_resolution` is True, all videos will be cropped and resized to the resolution of `args.max_height x args.max_width`. If `force_resolution` is False, `args.max_hxw` must be provided which determines the maximum token length of each video tensor.
- `gradient_checkpointing`: it is referred to MindSpore [recomputation](https://www.mindspore.cn/docs/en/r2.3.1/api_python/mindspore/mindspore.recompute.html) feature, which can save memory by recomputing the intermediate activations in the backward pass.
- `pretrained`: the pretrained checkpoint to be loaded as initial weights before training. If not provided, the OpenSoraT2V will use random initialization.
- `parallel_mode`: the parallelism mode chosen from ["data", "optim", "zero"], which denotes the data parallelism, the optimizer parallelism and the deepspeed zero_x parallelism.
- `zero_stage`: runs parallelism like deepspeed, supporting zero0, zero1, zero2, and zero3, if parallel_mode is "zero". By default, we use `--zero_stage 2` for all training stages.

For the stage 2 and stage 3 training scripts, please refer to `train_t2v_stage2.sh` and `train_t2v_stage3.sh`.

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
The edits allow to compute the loss on the validation set specified by `merge_data_val.txt` for every 1 epoch (defined by `val_interval`). `merge_data_val.txt` has the same format as `merge_data_train.txt`, but specifies a different subset from the train set. The validation loss will be recorded in the `result_val.log` under the output directory.

#### Sequence Parallelism

We also support training with sequence parallelism and zero2 parallelism together. This is enabled by setting `--sp_size`.

See `train_t2v_stage2.sh` under `scripts/text_condition/mult-devices/` for detailed usage.

#### Performance

We evaluated the training performance on Ascend NPUs. All experiments are running in PYNATIVE mode with MindSpore(2.3.1). The results are as follows.

| model name      | cards       |  stage     | batch size (global)   | video size  | Paramllelism |recompute |data sink | jit level| step time | train imgs/s |
|:----------------|:----------- |:---------:|:-----:|:----------:|:----------:|:----------:|:----------:|:----------:|-------------------:|:----------:|
| OpenSoraT2V_v1_3-2B/122 |  8   | 1 |  32  |    1x256x256     |         zero2                     | TRUE | FALSE | O0 |  4.37  | 7.32  |
| OpenSoraT2V_v1_3-2B/122 |  8   | 2 |  1  |    up to 93x640x640    |         zero2  + SP(sp_size=8)  |  TRUE | FALSE | O0 |  22.4s*  | 4.15 |
| OpenSoraT2V_v1_3-2B/122 |  8   | 3 |  8  |    93x352x640   |         zero2      |  TRUE | FALSE | O0 |  10.30  | 72.23 |

> SP: sequence parallelism.

> *: dynamic resolution using bucket sampler. The step time may vary across different batches due to the varied resolutions.

> train imgs/s: it is computed by $num\quad of\quad frames \times global\quad batch\quad size \div per\quad step\quad time$

## 👍 Acknowledgement
* [Latte](https://github.com/Vchitect/Latte): The **main codebase** we built upon and it is an wonderful video generated model.
* [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha): Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.
* [VideoGPT](https://github.com/wilson1yan/VideoGPT): Video Generation using VQ-VAE and Transformers.
* [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
* [FiT](https://github.com/whlzy/FiT): Flexible Vision Transformer for Diffusion Model.
* [Positional Interpolation](https://arxiv.org/abs/2306.15595): Extending Context Window of Large Language Models via Positional Interpolation.
