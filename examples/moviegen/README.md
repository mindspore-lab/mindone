# Movie Gen based on MindSpore

This repository implements the [Movie Gen](https://arxiv.org/abs/2410.13720) model presented by Meta.

Movie Gen is a family of foundation models that can natively generate high-fidelity images and videos
while also possessing the abilities to edit and personalize the videos.

We aim to explore an efficient implementation based on MindSpore and Ascend NPUs.
See our [report](docs/report.md) for more details.

## 📑 Development Plan

This project is in an early stage and under active development. We welcome the open-source community to contribute to
this project!

- Temporal Autoencoder (TAE)
    - [x] Inference
    - [x] Training
- Movie Gen 5B (T2I/V)
    - [x] Inference
    - [x] Training stage 1: T2I 256px
    - [x] Training stage 2: T2I/V 256px 256frames
    - [ ] Training stage 3: T2I/V 768px 256frames (under verification)
    - [x] Web Demo (Gradio)
- Movie Gen 30B (T2I/V)
    - [x] Inference
    - [x] Mixed parallelism training (support Ulysses-SP + ZeRO-3)
    - [x] Training stage 1: T2I 256px
    - [x] Training stage 2: T2V 256px 256frames
    - [ ] Training stage 3: T2I/V 768px 256frames
- Training with Buckets
    - [ ] Support variable resolutions and aspect ratios
    - [ ] Support variable number of frames
- Video Personalization (PT2V)
    - [ ] Inference
    - [ ] Training
- Video Editing
    - [ ] Inference
    - [ ] Training
- Video Super-Resolution
    - [ ] Inference
    - [ ] Training

## Demo

### 5B Model Stage 2

|                                                                                                                                                                                                                                                                                                                                         256x256x455                                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                                                                                                             256x256x455                                                                                                                                                                                                                                                                                                                                                              |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                                                                                                                                                                                                                               <video src="https://github.com/user-attachments/assets/84264678-a2c4-4605-93c7-4efce8b4647a" />                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                   <video src="https://github.com/user-attachments/assets/357c93f1-b129-4441-80dc-adbe0d089a3a" />                                                                                                                                                                                                                                                                                                                    |
| <details><summary>Caption</summary>The video showcases a person wearing a blue cap and a plaid shirt, sitting on the ground with a golden retriever dog. The person is seen engaging in an affectionate interaction with the dog, gently stroking its fur and at one point, caressing or scratching behind the dog's ears. Throughout the video, the dog remains relaxed and content, with its mouth slightly open as if panting or smiling. The setting is an outdoor grassy area with fallen leaves or twigs scattered on the ground, under warm lighting that creates a cozy, intimate atmosphere focused on the bonding moment between the person and their canine companion.</details> | <details><summary>Caption</summary>The video features a close-up view of a cat with striking blue eyes and a white furry face adorned with brown and black stripes on its head. Initially, the cat is seen looking directly at the camera with an attentive expression, held gently by a human hand around its neck area against a blurred indoor background with a brown surface. As the video progresses, the cat's gaze becomes more intense and focused, with its whiskers appearing more prominent and alert. The camera zooms in slightly, cropping out some of the surrounding area to bring the cat's face into closer view, maintaining the attentive and engaged demeanor of the feline throughout the sequence.</details> |
|                                                                                                                                                                                                                                                                                               <video src="https://github.com/user-attachments/assets/e89a6be6-1e5b-4508-8980-89d824824e34" />                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                   <video src="https://github.com/user-attachments/assets/70cdc452-cad8-4781-9975-1c9feb8b89d6" />                                                                                                                                                                                                                                                                                                                    |
|                                                              <details><summary>Caption</summary>The video showcases a static image of a bouquet of white roses, with the roses in various stages of bloom. The petals of the roses are delicate and pristine white, contrasting with the soft pink hues visible in their centers. The arrangement is full and lush, with stems protruding outwards. Throughout the video, there are no significant changes in the composition or positioning of the roses, and the background remains consistently blurred, ensuring the floral arrangement remains the focal point.</details>                                                              |                                      <details><summary>Caption</summary>The video showcases a majestic snow-capped mountain range against a cloudy sky, with the peaks covered in pristine white snow and jagged rocky outcrops protruding from the slopes. The mountains cast long shadows across the snow-covered terrain below. Initially, the sky is a vivid blue with wispy white clouds, but as the video progresses, the clouds become slightly more dispersed, revealing more of the blue sky. Throughout the video, the overall composition and grandeur of the mountain vistas remain consistent, maintaining the serene and awe-inspiring natural beauty of the landscape.</details>                                      |

### 30B Model Stage 2

|                                                                                                                                                                                                                                                                                                                              256x256x455                                                                                                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                  256x256x455                                                                                                                                                                                                                                  |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                                                                                                                                                                                                                    <video src="https://github.com/user-attachments/assets/e5558081-8710-4474-a522-a19a573a22e4" />                                                                                                                                                                                                                                                                                    |                                                                                                                                                                                        <video src="https://github.com/user-attachments/assets/d4625360-75f4-489a-893d-e4341b644be1" />                                                                                                                                                                                        |
| <details><summary>Caption</summary>The video showcases a serene aerial view of a mountainous landscape, consistently blanketed in snow and clouds throughout its duration. The foreground prominently features rugged, snow-capped peaks with jagged rock formations piercing through the pristine white snow. The background is consistently filled with a vast expanse of billowing clouds, interspersed with patches of blue sky above. The overall scene maintains a sense of tranquility and natural beauty, highlighting the grandeur of the mountainous terrain without any noticeable changes in the composition or perspective of the aerial view.</details> | <details><summary>Caption</summary>The video begins with a serene winter landscape featuring a frozen body of water in the foreground. The ice-covered surface is smooth and reflective, with patches of exposed water visible. In the background, a dense forest of evergreen trees lines the far shore, their branches covered in snow. The scene is hazy, with a grayish tint suggesting overcast or foggy conditions, maintaining a wintry ambiance throughout.</details> |  

## Requirements

<div align="center">

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

</div>

1. Install
   [CANN 8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1)
   and MindSpore according to the [official instructions](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```

## Model Weights

<details>
<summary><b>TAE</b></summary>

Download the TAE weights from
[here](https://download.mindspore.cn/toolkits/mindone/moviegen/tae_ucf101pt_mixkitft-b3b2e364.ckpt) and save them in the
`models/` directory.

</details>

<details>
<summary><b>Text Encoders</b></summary>

Downloading and conversion of the text encoders' weights to the `.safetensors` format can be done automatically by using
the following commands:

```shell
python tools/download_convert_st.py "google/byt5-small"
python tools/download_convert_st.py "google/ul2"
```

If you face an SSL certificate verification error, you can add `--disable_ssl_verify` option.

</details>

## Inference

### Generating Text Embeddings

Due to the large memory footprint of the text encoders, the inference and training pipelines don't support generating
text embeddings online. Therefore, you need to prepare them in advance by running the following command:

```shell
python scripts/inference_text_enc.py \
--model_name google/ul2 \
--prompts_file /path/to/prompts.csv \
--output_path /path/to/output/directory \
--model_max_length 512
```

> [!NOTE]
> We use the sequence length of 512 tokens for UL2, 256 for MetaCLIP, and 100 for ByT5.

### Text-to-Image

For more detailed instructions, please run `python scripts/inference.py --help`.

```shell
python scripts/inference.py \
--config configs/inference/moviegen_t2i_256px.yaml \
--model.name llama-5B \
--model.pretrained_model_path /path/to/llama-5B.ckpt \
--text_emb.ul2_dir /path/to/ul2_embeddings \
--text_emb.metaclip_dir /path/to/metaclip_embeddings \
--text_emb.byt5_dir /path/to/byt5_embeddings \
--image_size 256 455 \
--batch_size 2
```

### Text-to-Video

```shell
python scripts/inference.py \
--config configs/inference/moviegen_t2i_256px.yaml \
--model.name llama-5B \
--model.pretrained_model_path /path/to/llama-5B.ckpt \
--text_emb.ul2_dir /path/to/ul2_embeddings \
--text_emb.metaclip_dir /path/to/metaclip_embeddings \
--text_emb.byt5_dir /path/to/byt5_embeddings \
--image_size 256 455 \
--num_frames 32 \
--batch_size 2 \
--save_format mp4
```

### Gradio Demo

To launch the web demo, follow these steps:

1. Install Gradio:

```shell
pip install gradio
```

2. Run the demo script with the following configuration. The demo provides 80 pre-computed text prompts to choose from:

```shell
python scripts/gradio_demo.py \
--config configs/inference/moviegen_t2i_256px.yaml \
--model.name llama-5B \
--model.pretrained_model_path /path/to/llama-5B.ckpt \
--text_emb.ul2_dir /path/to/ul2-embedding.ckpt \
--text_emb.metaclip_dir /path/to/metaclip-embedding.ckpt \
--text_emb.byt5_dir /path/to/byt5-embedding.ckpt \
--image_size 256 455  
--num_frames 32  
--save_format mp4
```

Note: Make sure to replace the `/path/to/` placeholders with your actual model and embedding paths.

## Training

Movie Gen is trained jointly on images and videos in 4 stages:

1. Training on images at 256 px resolution.
2. Joint training on images and videos at 256 px resolution.
3. Joint training at 768 px resolution.
4. Fine-tune the model on high quality videos.

Images are treated as single frame videos, enabling the use of the same model to generate both images and videos.
Compared to video data, paired image-text datasets are easier to scale with diverse concepts and styles,
and thus joint modeling of image and video leads to better generalization.

To train Movie Gen, run the following commands:

```shell
scripts/moviegen/stage1_train.sh  # for stage 1 training
scripts/moviegen/stage2_train.sh  # for stage 2 training
scripts/moviegen/stage3_train.sh  # for stage 3 training (currently under verification)
```

### Dataset Preparation

Paths to videos and their corresponding captions should be stored in a CSV file with two columns: `video` and `caption`.
For example:

```text
video,caption
video_folder/part01/vid001.mp4,a cartoon character is walking through
video_folder/part01/vid002.mp4,a red and white ball with an angry look on its face
```

### Generating Text Embeddings

Due to the large memory footprint of the text encoders, the inference and training pipelines don't support generating
text embeddings online. Please refer to the [Generating Text Embeddings](#generating-text-embeddings) section under the
Inference section for details.

### Cache Video Embedding (Optional)

If you have sufficient storage budget, you can cache the video embeddings to speed up training by using the following
command:

```shell
python scripts/inference_tae.py \
--tae.pretrained=/path/to/tae.ckpt \
--tae.dtype=bf16 \
--video_data.folder=/path/to/folder/with/videos/ \
--output_path=/path/to/output/directory/ \
--video_data.size=256 \
--video_data.crop_size=[256,455]
```

### Performance

Experiments were conducted on Ascend 910* using MindSpore 2.3.1 in Graph mode.

> [!NOTE]
> We trained all the models using BF16 precision and JIT level `O1`.  
> To disable Flash Attention recompute, set `model.not_recompute_fa` to `True`.

| Model | Cards |   Stage   |      Batch size       |       Resolution        | Compile time |        Recompute         | Gradient Acc | ZeRO | Sequence Parallel | TAE Cache | Time (s/step) |                             Config                             |
|:-----:|:-----:|:---------:|:---------------------:|:-----------------------:|:------------:|:------------------------:|:------------:|:----:|:-----------------:|:---------:|:-------------:|:--------------------------------------------------------------:|
|  30B  |   8   |  1 (T2I)  |          10           |         256x455         |      6m      |            ON            |      1       |  3   |        No         |    Yes    |     5.14      |  [stage1_t2i_256px.yaml](configs/train/stage1_t2i_256px.yaml)  |
|  30B  |   8   |  2 (T2V)  |       Video: 1        |       256x256x455       |      7m      |            ON            |      1       |  3   |     8 shards      |    Yes    |     4.04      | [stage2_t2iv_256px.yaml](configs/train/stage2_t2iv_256px.yaml) |
|  30B  |   8   |  3 (T2V)  |       Video: 1        |      256x576x1024       |      7m      |            ON            |      1       |  3   |     8 shards      |    Yes    |     37.7      | [stage3_t2iv_768px.yaml](configs/train/stage3_t2iv_768px.yaml) |
|  5B   |   8   |  1 (T2I)  |          10           |         256x455         |      3m      |           OFF            |      1       |  3   |        No         |    Yes    |     0.82      |  [stage1_t2i_256px.yaml](configs/train/stage1_t2i_256px.yaml)  |
|  5B   |   8   | 2 (T2I/V) | Image: 1<br/>Video: 1 | 256x455<br/>256 frames  |      3m      | ON<br/>(No FA recompute) |      5       |  2   |        No         |    Yes    |     4.12      | [stage2_t2iv_256px.yaml](configs/train/stage2_t2iv_256px.yaml) |
|  5B   |   8   | 3 (T2I/V) | Image: 1<br/>Video: 1 | 576x1024<br/>256 frames |    5m 40s    |            ON            |      5       |  2   |        No         |    Yes    |     83.2      | [stage3_t2iv_768px.yaml](configs/train/stage3_t2iv_768px.yaml) |
|  1B   |   8   |  1 (T2I)  |          10           |         256x455         |      2m      |           OFF            |      1       |  No  |        No         |    Yes    |     0.32      |  [stage1_t2i_256px.yaml](configs/train/stage1_t2i_256px.yaml)  |
|  1B   |   8   | 2 (T2I/V) | Image: 1<br/>Video: 1 | 256x455<br/>256 frames  |    2m 20s    |           OFF            |      5       |  No  |        No         |    Yes    |     2.12      | [stage2_t2iv_256px.yaml](configs/train/stage2_t2iv_256px.yaml) |
|  1B   |   8   | 3 (T2I/V) | Image: 1<br/>Video: 1 | 576x1024<br/>256 frames |    2m 30s    | ON<br/>(No FA recompute) |      5       |  No  |        No         |    Yes    |     23.2      | [stage3_t2iv_768px.yaml](configs/train/stage3_t2iv_768px.yaml) |

### Validation During Training

Validation can be enabled by either setting parameters in the `valid` field of the configuration file
([example](configs/train/stage1_t2i_256px.yaml)) or by supplying the following arguments to `train.py`:

```shell
--valid.sampling_steps 10 \
--valid.frequency 100 \
--valid.dataset.csv_path /path/to/valid_dataset.csv \
--valid.dataset.video_folder /path/to/videos \
--valid.dataset.text_emb_folder.ul2 /path/to/ul2_embeddings \
--valid.dataset.text_emb_folder.metaclip /path/to/metaclip_embeddings \
--valid.dataset.text_emb_folder.byt5 /path/to/byt5_embeddings
```

## Evaluation

Coming soon.

## TAE Training & Evaluation

### Dataset Preparation

We need to prepare a csv annotation file listing the path to each input video related to the root folder, indicated by
the `video_folder` argument. An example is

```
video
dance/vid001.mp4
dance/vid002.mp4
dance/vid003.mp4
...
```

Taking UCF-101, for example, please download the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset and extract
it to `datasets/UCF-101` folder.

### Training

TAE is trained to optimize the reconstruction loss, perceptual loss, and the outlier penalty loss (OPL) proposed in the
MovieGen paper.

To launch training, please run

```shell
python scripts/train_tae.py \
--config configs/tae/train/mixed_256x256x32.yaml \
--output_path /path/to/save_ckpt_and_log \
--csv_path /path/to/video_train.csv  \
--folder /path/to/video_root_folder  \
```

Unlike the paper, we found that OPL loss doesn't benefit the training outcome in our ablation study (with OPL, PSNR is
31.17). Thus, we disable OPL loss by default. You may enable it by appending `--use_outlier_penalty_loss True`.

For more details on the arguments, please run `python scripts/train_tae.py --help`

### Evaluation

To run video reconstruction with the trained TAE model and evaluate the PSNR and SSIM on the test set, please run

```shell
python scripts/eval_tae.py \
--ckpt_path /path/to/tae.ckpt \
--batch_size 2 \
--num_frames 32  \
--image_size 256 \
--csv_path  /path/to/video_test.csv  \
--folder /path/to/video_root_folder  \
```

The reconstructed videos will be saved in `samples/recons`.

### Performance

Here, we report the training performance and evaluation results on the UCF-101 dataset.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name | cards | batch size | resolution | precision | jit level | graph compile | s/step | PSNR  | SSIM |                      recipe                       |
|:----------:|:-----:|:----------:|:----------:|:---------:|:---------:|:-------------:|:------:|:-----:|:----:|:-------------------------------------------------:|
|    TAE     |   1   |     1      | 256x256x32 |   bf16    |    O0     |     2 min     |  2.18  | 31.35 | 0.92 | [config](configs/tae/train/mixed_256x256x32.yaml) |
