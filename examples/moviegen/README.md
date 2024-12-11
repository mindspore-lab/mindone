# Movie Gen

This repository implements the [Movie Gen](https://arxiv.org/abs/2410.13720) model presented by Meta.

Movie Gen is a family of foundation models that can natively generate high-fidelity images and videos
while also possessing the abilities to edit and personalize the videos.

Meta researchers found that scaling the training data, compute, and model parameters of a simple
Transformer-based ([LLaMa3](https://arxiv.org/abs/2407.21783)) model trained with
[Flow Matching](https://arxiv.org/abs/2210.02747) yields high quality generative models for video or audio.

### Features

1. :white_check_mark: Text-to-Video synthesis
2. \[Coming soon] Video personalization
3. \[Coming soon] Video editing

<details>
<summary>TODO</summary>

- [ ] Fix EMA.
- [ ] Use ByT5 for encoding visual text only (i.e., text within quotes).
- [ ] CFG inference.
- [ ] Multi-aspect and variable length video training (including PE interpolation).
- [ ] Fix Model Parallel training.
- [ ] Add FPS conditioning.

</details>

## Demo

Coming soon.

## Architecture

<details>
<summary><b>Architecture details</b></summary>

### Transformer Backbone

The Movie Gen family of models contains the following variations: 1B, 5B, and 30B parameters.
It uses the [LLaMa3](https://arxiv.org/abs/2407.21783) backbone architecture for the joint image-video generation model,
enabling confident scaling of the model size while maintaining efficient training.

There are three changes to the LLaMa3 Transformer block for the use case of video generation using Flow Matching:

1. Add a cross-attention module between the self-attention module and the feed forward network (FFN)
   to each Transformer block to incorporate text conditioning based on the text prompt embedding **P**.
   Multiple different text encoders are leveraged due to their complementary strengths
   (see [Text Encoders](#text-encoders)).
2. Add adaptive layer norm blocks to incorporate the time-step t to the Transformer, as used in prior work
   ([DiT](https://arxiv.org/abs/2212.09748)).
3. Use full bidirectional attention instead of causal attention used in language modeling.

### TAE

[//]: # (TODO)

### Text Encoders

Movie Gen uses a combination of [UL2](https://arxiv.org/abs/2205.05131), [ByT5](https://arxiv.org/abs/2105.13626), and
Long-prompt [MetaCLIP](https://arxiv.org/abs/2309.16671) as text encoders to provide both semantic-level and
character-level text understanding for the backbone:

- **UL2** is trained using massive text-only data and potentially provides strong text reasoning abilities in its
  features.
- **Long-prompt MetaCLIP** provides text representations that are aligned with visual representations that are
  beneficial
  for cross-modal generation.
- **ByT5** encoder is only used to encode visual text, i.e., the part of the text prompt that explicitly asks for a
  character string to be generated in the output image / video.

</details>

## Installation

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

1. Install MindSpore according to the [official instructions](https://www.mindspore.cn/install).
   For Ascend devices, please install
   [CANN8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1)
   and [MindSpore 2.3.1](https://www.mindspore.cn/install).
2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```

## Model Weights

<details>
<summary><b>TAE</b></summary>

We use SD3.5 VAE to initialize the spatial layers of TAE since both have the same number of latent channels, i.e., 16.

1. Download SD3.5 VAE from [huggingface](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main/vae)

2. Inflate VAE checkpoint for TAE initialization by
    ```shell
    python inflate_vae_to_tae.py --src /path/to/sd3.5_vae/diffusion_pytorch_model.safetensors --target models/tae_vae2d.ckpt
    ```

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

## Generating Text Embeddings

Due to the large memory footprint of the text encoders, the inference and training pipelines don't support generating
text embeddings online. Therefore, you need to prepare them in advance by running the following command:

```shell
python inference_text_enc.py \
--model_name google/ul2 \
--prompts_file /path/to/prompts.csv \
--output_path /path/to/output/directory \
--model_max_length 512
```

> [!NOTE]
> We use the sequence length of 512 tokens for UL2, 256 for MetaCLIP, and 100 for ByT5.

## Inference

For more detailed instructions, please run `python inference.py --help`.

### Text-to-Image

```shell
python inference.py \
--config configs/inference/moviegen_t2i_256x256.yaml \
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
python inference.py \
--config configs/inference/moviegen_t2i_256x256.yaml \
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

### TAE

#### Encoding Video

```python
from mg.models.tae import TemporalAutoencoder

# may set use_tile=True to save memory
tae = TemporalAutoencoder(
    pretrained='/path/to/tae.ckpt',
    use_tile=False,
)

# x - a batch of videos, shape (b c t h w)
z, _, _ = tae.encode(x)

# you may scale z by:
z = (z - tae.shift_factor) * tae.scale_factor
```

For detailed arguments, please refer to the docstring in [tae.py](mg/models/tae/tae.py)

#### Decoding Video Latent

```python
# if z is scaled, you should unscale at first:
z = z / tae.scale_factor + tae.shift_factor

# z - a batch of video latent, shape (b c t h w)
x = tae.decode(z)

# for image decoding, set num_target_frames to discard the spurious frames
x = tae.decode(z, num_target_frames=1)
```

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
scripts/stage1_train.sh # for stage 1 training
scripts/stage2_train.sh # for stage 2 training
```

### Dataset Preparation

Paths to videos and their corresponding captions should be stored in a CSV file with two columns: `video` and `caption`.
For example:

```text
video,caption
video_folder/part01/vid001.mp4,a cartoon character is walking through
video_folder/part01/vid002.mp4,a red and white ball with an angry look on its face
```

### Cache Video Embedding (Optional)

If you have sufficient storage budget, you can cache the video embeddings to speed up training by using the following
command:

```shell
python inference_tae_enc.py \
--tae.pretrained=/path/to/tae.ckpt \
--tae.dtype=bf16 \
--data.folder=/path/to/folder/with/videos/ \
--output_path=/path/to/output/directory/ \
--data.size=256 \
--data.crop_size=[256,455]
```

### Performance

| Model |      Context      | Jit level |   Stage   | Precision |          Resolution          | TAE Cache |       Batch size        | NPUs | Time (s/step) |                              Config                               |
|:-----:|:-----------------:|:---------:|:---------:|:---------:|:----------------------------:|:---------:|:-----------------------:|:----:|:-------------:|:-----------------------------------------------------------------:|
|  5B   | D910*-C18-MS2.3.1 |    O1     |  1 (T2I)  |   BF16    |        256x455 (16:9)        |    No     |           20            |  4   |     4.47      | [stage1_t2i_256x256.yaml](configs/train/stage1_t2i_256x256.yaml)  |
|  5B   | D910*-C18-MS2.3.1 |    O0     | 2 (T2I/V) |   BF16    | 256x455 (16:9)<br/>32 frames |    No     | Image: 10<br/>Video: 5  |  8   |     5.26      | [stage1_t2i_256x256.yaml](configs/train/stage2_t2iv_256x256.yaml) |
|  1B   | D910*-C18-MS2.3.1 |    O1     |  1 (T2I)  |   BF16    |        256x455 (16:9)        |    Yes    |           10            |  8   |     0.53      | [stage1_t2i_256x256.yaml](configs/train/stage1_t2i_256x256.yaml)  |
|  1B   | D910*-C18-MS2.3.1 |    O0     | 2 (T2I/V) |   BF16    | 256x455 (16:9)<br/>32 frames |    Yes    | Image: 10<br/>Video: 10 |  8   |     2.08      | [stage1_t2i_256x256.yaml](configs/train/stage2_t2iv_256x256.yaml) |

### Validation During Training

Validation can be enabled by either setting parameters in the `valid` field of the configuration file
([example](configs/train/stage1_t2i_256x256.yaml)) or by supplying the following arguments to `train.py`:

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
python train_tae.py \
--config configs/tae/train/mixed_256x256x32.yaml \
--output_path /path/to/save_ckpt_and_log \
--csv_path /path/to/video_train.csv  \
--folder /path/to/video_root_folder  \
```

Different from the paper, we found that OPL loss doesn't benefit the training outcome in our ablation study (reducing in
lower PSNR decreased). Thus, we disable OPL loss by default. You may enable it by appending
`--use_outlier_penalty_loss True`

For more details on the arguments, please run `python scripts/train_tae.py --help`

### Evaluation

To run video reconstruction with the trained TAE model and evaluate the PSNR and SSIM on the test set, please run

```shell
python eval_tae.py \
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
