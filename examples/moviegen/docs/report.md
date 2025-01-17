# MindSpore Movie Gen Report

[Movie Gen](https://ai.meta.com/static-resource/movie-gen-research-paper) is a family of foundation models that can
natively generate high-fidelity images, videos, and audio. Meta researchers found that scaling the training data,
compute, and model parameters of the transformer-based ([LLaMa3](https://arxiv.org/abs/2407.21783)) model trained
with [Flow Matching](https://arxiv.org/abs/2210.02747) yields high-quality generative models for video or audio.

Movie Gen supports text-to-video/image generation (MovieGenVideo), video personalization  (PersonalizedMovieGen), and
video editing  (MovieGenEdit).

In this report, we will focus on MovieGenVideo and explore how to implement it with MindSpore, enabling model scaling
and training efficiency.

At this moment, we support training MovieGenVideo with the following configuration.

| model scale | image | 256px @256 | 768px @256 |
|-------------|-------|------------|------------|
| 1B          | ✅     | ✅          | ✅          |
| 5B          | ✅     | ✅          | 🆗         |
| 30B         | 🆗    | ✅          | TODO       |

Here ✅ means that training accuracy has been verified on a small-scale dataset, and 🆗 means training is supported, but
the accuracy is under verification.

## Temporal Autoencoder (TAE)

TAE is used to encode the RGB pixel-space videos and images into a spatio-temporally compressed latent space. In
particular, the input is compressed by 8x across each spatial dimension H and W, and the temporal dimension T. We follow
the framework of Meta Movie Gen [[1](#references)] as below.

<p align="center"><img width="700" alt="TAE Framework" src="https://github.com/user-attachments/assets/678c2ce6-28b8-4bda-b8a3-fac921595b8a"/>
<br><em> Figure 1. Video Encoding and Decoding using TAE </em></p>

TAE inflates an image autoencoder by adding 1-D temporal convolution in resnet blocks and attention blocks. Temporal
compression is done by injecting temporal downsample and upsample layers.

### Key design & implementation

In this section, we explore the design and implementation details not illustrated in the Movie Gen paper. For example,
how to perform padding and initialization for the Conv 2.5-D layers and how to configure the training frames.

#### SD3.5 VAE as the base image encoder

In TAE, the number of channels of the latent space is 16 (C=16). It can help improve both the reconstruction and the
generation performance compared to C=4 used in OpenSora or SDXL vae.

We choose to use the [VAE]() in Stable Diffusion 3.5 as the image encoder to build TAE for it has the same number of
latent channels and can generalize well in image generation.

#### Conv2.5d implementation

Firstly, we replace the Conv2d in VAE with Conv2.5d, which consists of a 2D spatial convolution followed by a 1D
temporal convolution.

For 1D temporal convolution, we set kernel size 3, stride 1, symmetric replicate padding with padding size (1, 1), and
input/output channels the same as spatial conv. We initialize the kernel weight to preserve the spatial features
(i.e., preserve image encoding after temporal initialization). Therefore, we propose to use `centric` initialization as
illustrated below.

```python
w = self.conv_temp.weight
ch = int(w.shape[0])
value = np.zeros(tuple(w.shape))
for i in range(ch):
    value[i, i, 0, 1] = 1
w.set_data(ms.Tensor(value, dtype=ms.float32))
```

#### Temporal Downsampling

Paper: "Temporal downsampling is performed via strided convolution with a stride of 2".

Our implementation: the strided convolution is computed using conv1d of kernel size 3, stride 2, and symmetric replicate
padding. `centric` initialization (as mentioned in the above conv2.5 section) is used to initialize the conv kernel
weight.

To achieve 8x temporal compression, we apply 3 temporal downsampling layers, each placed after the spatial downsampling
layer in the first 3 levels.

#### Temporal Upsampling

Paper: "upsampling by nearest-neighbor interpolation followed by convolution"

Our design:

1. nearest-neighbour interpolation along the temporal dimension
2. conv1d: kernel size 3, stride 1, symmetric replicate padding, and `centric` initialization.

To achieve 8x temporal compression, we apply 3 temporal upsampling layers, each placed after the spatial upsampling
layer of the last 3 levels.

### Evaluation

We conduct experiments to verify our implementation's effectiveness on
the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset containing 13,320 videos. We split the videos into
training and test sets by 8:2.

The training performance on MindSpore 2.3.1 and Ascend 910* and the accuracy on the test set are as follows.

| model name | cards | batch size | resolution | precision | OPL Loss | s/step | PSNR  | SSIM |
|:----------:|:-----:|:----------:|:----------:|:---------:|:--------:|:------:|:-----:|:----:|
|    TAE     |   1   |     1      | 256x256x32 |   bf16    |   OFF    |  2.18  | 31.35 | 0.92 |
|    TAE     |   1   |     1      | 256x256x32 |   bf16    |    ON    |  2.18  | 31.17 | 0.92 |

The hyperparameters we used are as follows.

```yaml
kl loss weight: 1.0e-06
perceptual and reconstruction loss weight: 1.0
outlier penalty loss weight: 1.0
optimizer: adamw
learning rate: 1e-5
```

Here is the comparison between the origin videos (left) and the videos reconstructed with the trained TAE model (right).


<p float="center">
<img src=https://github.com/user-attachments/assets/ba3362e4-2210-4811-bedf-f19316f511d3 width="45%" />
<img src=https://github.com/user-attachments/assets/36257aef-72f0-4f4f-8bd3-dc8fb0a33fd8 width="45%" />
</p>

We further fine-tune the TAE model on the mixkit dataset, a high-quality video dataset in 1080P resolution. Here are the
results.

<p float="center">
<img src=https://github.com/user-attachments/assets/7978489b-508b-4204-a4d7-d11dda3f905c width="45%" />
<img src=https://github.com/user-attachments/assets/e87105d9-1ff1-4a4c-bbfb-e07615f0fe6d width="45%" />
</p>

The fine-tuned TAE is then used in MovieGenVideo transformer training as shown below.

## Diffusion Transformer

### Architecture

MovieGenVideo uses the [LLaMa3](https://arxiv.org/abs/2407.21783) backbone architecture for the joint image-video
generation
model, enabling confident scaling of the model size while maintaining efficient training, as shown in the figure below.

<p align="center">
<img alt="Transformer backbone and model parallelism" width="700" src="https://github.com/user-attachments/assets/87811e4f-5e49-4530-b43f-734782bf0e0a"/>
<br><em>Figure 2. Transformer backbone and model parallelism</em>
</p>

There are three changes to the LLaMa3 Transformer block for the use case of video generation using Flow Matching:

1. Add a cross-attention module between the self-attention module and the feed-forward network (FFN)
   to each Transformer block to incorporate text conditioning based on the text prompt embedding **P**.
   Multiple different text encoders are leveraged due to their complementary strengths
   (see [Text Encoders](#text-encoders)).
2. Add adaptive layer norm blocks to incorporate the time-step $t$ to the Transformer, as used in prior work
   ([DiT](https://arxiv.org/abs/2212.09748)).
3. Use full bidirectional attention instead of causal attention used in language modeling.

We have implemented the MovieGenVideo architecture in the following variations: 1B, 5B, and 30B parameters.

| Model | Layers | Model Dimension | FFN Dimension | Attention Heads |
|:-----:|:------:|:---------------:|:-------------:|:---------------:|
|  1B   |   24   |      1536       |     4096      |       16        |
|  5B   |   32   |      3072       |     8192      |       24        |
|  30B  |   48   |      6144       |     16384     |       48        |

Detailed code implementation can be referred to:
[LLaMa3 Backbone](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/network.py#L273),
[Transformer Block](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/network.py#L52).

### Sequence Parallelism

The official [Movie Gen](https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models/) employs
3D parallelism to enable model-level scaling across three dimensions: the number of parameters, input tokens, and
dataset size, while also allowing horizontal scale-out to additional NPUs. It leverages a combination
of [fully sharded data parallelism](https://arxiv.org/abs/2304.11277), [tensor parallelism](https://arxiv.org/abs/1909.08053), [sequence parallelism](https://arxiv.org/abs/2205.05198),
and [context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html).

Inspired by recent developments in long-sequence parallelism ([Ulysses-SP](https://arxiv.org/abs/2309.14509)
and [USP](https://arxiv.org/abs/2405.07719)), we implement model parallelism
using [Ulysses-SP](https://arxiv.org/abs/2309.14509) together with [ZeRO-3](https://arxiv.org/abs/1910.02054),instead of
the approach used in Movie Gen. Ulysses-SP utilizes `All2ALL` communication for segments of the QKV tensors, drastically
reducing communication costs compared to sequence parallelism implemented
in [Megatron-LM](https://arxiv.org/abs/2405.07719), as well as the sequence
parallelism mentioned
in [Movie Gen](https://ai.meta.com/research/publications/movie-gen-a-cast-of-media-foundation-models/). Alongside
ZeRO-3, it achieves similar memory efficiency to [Megatron-LM](https://arxiv.org/abs/2405.07719). Experimental results
show that using Ulysses-SP + ZeRO-3, we can train a model of similar scale compared to 3D parallelism, with over 2x
speed boost in training, corroborating the findings
in [Ulysses-SP](https://arxiv.org/abs/2309.14509), [USP](https://arxiv.org/abs/2405.07719)
and [DSP](https://arxiv.org/abs/2403.10266).

### Training Details

Movie Gen is trained jointly on images and videos. Images are treated as single-frame videos, enabling the use of the
same model to generate both images and videos. Compared to video data, paired image-text datasets are easier to scale
with diverse concepts and styles, and thus joint modeling of image and video leads to better generalization.

Training is performed in multiple stages for better efficiency:

- Stage 1: Text-to-image pre-raining on 256 px images.
- Stage 2: T2I/V joint training on low-resolution images and videos of 256 px.
  Following the paper, we double the spatial [PE](#learnable-positional-embedding-pe) layers to accommodate
  various aspect ratios, add new temporal PE layers to support up to 32 latent frames and initialize spatial PE layers
  from the T2I model with 2x expansion.
- Stage 3: T2I/V joint training on high-resolution images and videos of 768 px.  
  For this stage, we expand the spatial PE layers by 3x.

#### Training Objective

Following the paper, we trained the transformer with [Flow Matching](https://arxiv.org/abs/2210.02747) with a simple
linear interpolation scheme.
It is trained to predict the velocity $V_t = \frac{dX_t}{dt}$ which teaches it to 'move' the sample $X_t$
in the direction of the video sample $X_1$. The ground truth velocity is derived by:
$$V_t = X_1 - (1-\sigma_{min})X_0$$. Note that this simple interpolation scheme naturally ensures zero terminal SNR
at $t=0$.

#### Learning Rate Scheduling

We decrease the learning rate by half whenever the validation loss plateaus to continue improving the model performance.

#### Validation During Training

As pointed out in the paper, the validation loss is well correlated with human evaluation results as the later
checkpoints with lower validation loss perform better in the human evaluations. This suggests that the flow-matching
validation loss can serve as a useful proxy for evaluations during model development. Similar observation was made by
the authors of [SD3](https://arxiv.org/abs/2403.03206). For this reason, we maintain a validation set of unseen videos
and monitor the validation loss throughout training.

### Bucketization for variable duration and size (under verification)

To support training with diverse video lengths and aspect ratios, we have integrated the data bucketing feature
in [hpcai-opensora](https://github.com/mindspore-lab/mindone/tree/master/examples/opensora_hpcai#multi-resolution-training).
This feature is under verification.

### Inference Details

Movie Gen uses a simple first-order Euler ODE solver with a unique t-schedule tailored to the model. Specifically, the
quality of an N-step video generation process can be closely approximated with merely 50 steps by implementing a
**linear-quadratic t-schedule**.

<p align="center">
<img alt="The linear-quadratic t-schedule" height="250" src="https://github.com/user-attachments/assets/888080ac-b162-4de0-a420-b2fb00c66fff"/>
<br><em>Figure 3. The linear-quadratic t-schedule</em>
</p>

This approach follows the first 25 steps of an $N$-step linear schedule and then approximates the remaining $N-25$ steps
with 25 quadratically placed steps. The linear-quadratic strategy is predicated on the observation that the first
inference steps are pivotal in setting up the scene and motion of the video since most changes occur in the first
solver steps.

Our implementation can be referred
to [here](https://github.com/hadipash/mindone/blob/movie_gen/examples/moviegen/mg/schedulers/rectified_flow.py#L55-L61)

[//]: # (TODO: fix the link above)

### Evaluation

To verify the effectiveness of our design and implementation, we perform 3-stage training on
a [mixkit](https://mixkit.co/) subset consisting of 100 HQ videos up to 1080P.

Experiments were conducted on Ascend 910* using MindSpore 2.3.1 in graph mode.

| Model | Cards |   Stage   |      Batch size       |       Resolution        |        Recompute         | TAE Cache | Sequence Parallel | Time (s/step) |                              Recipe                               |
|:-----:|:-----:|:---------:|:---------------------:|:-----------------------:|:------------------------:|:---------:|:-----------------:|:-------------:|:-----------------------------------------------------------------:|
|  30B  |   8   |  1 (T2I)  |          10           |         256x455         |            ON            |    ON     |        NO         |     5.14      |  [stage1_t2i_256px.yaml](../configs/train/stage1_t2i_256px.yaml)  |
|  30B  |   8   |  2 (T2V)  |       Video: 1        |       256x256x455       |            ON            |    ON     |     8 shards      |     4.04      | [stage2_t2iv_256px.yaml](../configs/train/stage2_t2iv_256px.yaml) |
|  30B  |   8   |  3 (T2V)  |       Video: 1        |      256x576x1024       |            ON            |    ON     |     8 shards      |     37.7      | [stage3_t2iv_768px.yaml](../configs/train/stage3_t2iv_768px.yaml) |
|  5B   |   8   |  1 (T2I)  |          10           |         256x455         |           OFF            |    ON     |        NO         |     0.82      |  [stage1_t2i_256px.yaml](../configs/train/stage1_t2i_256px.yaml)  |
|  5B   |   8   | 2 (T2I/V) | Image: 1<br/>Video: 1 | 256x455<br/>256 frames  | ON<br/>(No FA recompute) |    ON     |        NO         |     4.12      | [stage2_t2iv_256px.yaml](../configs/train/stage2_t2iv_256px.yaml) |
|  5B   |   8   | 3 (T2I/V) | Image: 1<br/>Video: 1 | 576x1024<br/>256 frames |            ON            |    ON     |        NO         |     83.2      | [stage3_t2iv_768px.yaml](../configs/train/stage3_t2iv_768px.yaml) |
|  1B   |   8   |  1 (T2I)  |          10           |         256x455         |           OFF            |    ON     |        NO         |     0.32      |  [stage1_t2i_256px.yaml](../configs/train/stage1_t2i_256px.yaml)  |
|  1B   |   8   | 2 (T2I/V) | Image: 1<br/>Video: 1 | 256x455<br/>256 frames  |           OFF            |    ON     |        NO         |     2.12      | [stage2_t2iv_256px.yaml](../configs/train/stage2_t2iv_256px.yaml) |
|  1B   |   8   | 3 (T2I/V) | Image: 1<br/>Video: 1 | 576x1024<br/>256 frames | ON<br/>(No FA recompute) |    ON     |        NO         |     23.2      | [stage3_t2iv_768px.yaml](../configs/train/stage3_t2iv_768px.yaml) |

> [!NOTE]
> All the models are trained with BF16 precision.

#### Detailed Training Scripts

##### Stage 1

<details>
<summary>Shell script</summary>

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

# log level
export GLOG_v=2

stage1_dir=output/stage1_t2i_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$stage1_dir"  \
python scripts/train.py \
  --config configs/train/stage1_t2i_256px.yaml \
  --env.mode 0 \
  --env.jit_level O1 \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --train.settings.zero_stage 3 \
  --model.recompute_every_nth_block "" \
  --dataset.csv_path ../../datasets/mixkit-100videos/video_caption_train.csv \
  --dataset.video_folder ../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../datasets/mixkit-100videos/tae_latent_images \
  --dataset.text_emb_folder.ul2 ../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.deterministic_sample True \
  --dataloader.batch_size 10 \
  --valid.dataset "" \
  --train.ema "" \
  --train.optimizer.weight_decay 0 \
  --train.save.ckpt_save_policy latest_k \
  --train.steps 2000 \
  --train.output_path "$stage1_dir"
```

</details>

##### Stage 2

<details>
<summary>Shell script</summary>

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

stage2_dir=output/stage2_t2iv_256px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$stage2_dir"  \
python scripts/train.py \
  --config configs/train/stage2_t2iv_256px.yaml \
  --env.mode 0 \
  --env.jit_level O1 \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.pretrained_model_path "$stage1_dir"/ckpt/llama-5B-s2000.ckpt\
  --train.settings.zero_stage 2 \
  --model.not_recompute_fa True \
  --dataset.csv_path ../../datasets/mixkit-100videos/video_caption_train.csv \
  --dataset.video_folder ../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../datasets/mixkit-100videos/tae_latent \
  --dataset.text_emb_folder.ul2 ../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.deterministic_sample True \
  --dataloader.batch_size.image_batch_size 1 \
  --dataloader.batch_size.video_batch_size 1 \
  --train.ema "" \
  --train.lr_scheduler.lr 0.0001 \
  --train.optimizer.weight_decay 0 \
  --train.settings.gradient_accumulation_steps 5 \
  --train.steps 40000 \
  --train.output_path "$stage2_dir"
```

</details>

##### Stage 3

<details>
<summary>Shell script</summary>

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

stage3_dir=output/stage3_t2iv_768px/$(date +"%Y.%m.%d-%H.%M.%S")

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir="$stage3_dir"  \
python scripts/train.py \
  --config configs/train/stage3_t2iv_768px.yaml \
  --env.mode 0 \
  --env.jit_level O1 \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.pretrained_model_path "$stage2_dir"/ckpt/llama-5B-s40000.ckpt\
  --train.settings.zero_stage 2 \
  --dataset.csv_path ../../datasets/mixkit-100videos/video_caption_train.csv \
  --dataset.video_folder ../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../datasets/mixkit-100videos/high_tae_latent \
  --dataset.text_emb_folder.ul2 ../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataset.deterministic_sample True \
  --dataloader.batch_size.image_batch_size 1 \
  --dataloader.batch_size.video_batch_size 1 \
  --train.ema "" \
  --train.optimizer.weight_decay 0 \
  --train.settings.gradient_accumulation_steps 5 \
  --train.steps 30000 \
  --train.output_path "$stage3_dir"
```

</details>

### Generated Video Examples

#### 5B Model Stage 2

|                                                                                                                                                                                                                                                                                                                                         256x256x455                                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                                                                                                             256x256x455                                                                                                                                                                                                                                                                                                                                                              |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                                                                                                                                                                                                                               <video src="https://github.com/user-attachments/assets/84264678-a2c4-4605-93c7-4efce8b4647a" />                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                   <video src="https://github.com/user-attachments/assets/357c93f1-b129-4441-80dc-adbe0d089a3a" />                                                                                                                                                                                                                                                                                                                    |
| <details><summary>Caption</summary>The video showcases a person wearing a blue cap and a plaid shirt, sitting on the ground with a golden retriever dog. The person is seen engaging in an affectionate interaction with the dog, gently stroking its fur and at one point, caressing or scratching behind the dog's ears. Throughout the video, the dog remains relaxed and content, with its mouth slightly open as if panting or smiling. The setting is an outdoor grassy area with fallen leaves or twigs scattered on the ground, under warm lighting that creates a cozy, intimate atmosphere focused on the bonding moment between the person and their canine companion.</details> | <details><summary>Caption</summary>The video features a close-up view of a cat with striking blue eyes and a white furry face adorned with brown and black stripes on its head. Initially, the cat is seen looking directly at the camera with an attentive expression, held gently by a human hand around its neck area against a blurred indoor background with a brown surface. As the video progresses, the cat's gaze becomes more intense and focused, with its whiskers appearing more prominent and alert. The camera zooms in slightly, cropping out some of the surrounding area to bring the cat's face into closer view, maintaining the attentive and engaged demeanor of the feline throughout the sequence.</details> |
|                                                                                                                                                                                                                                                                                               <video src="https://github.com/user-attachments/assets/e89a6be6-1e5b-4508-8980-89d824824e34" />                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                   <video src="https://github.com/user-attachments/assets/70cdc452-cad8-4781-9975-1c9feb8b89d6" />                                                                                                                                                                                                                                                                                                                    |
|                                                              <details><summary>Caption</summary>The video showcases a static image of a bouquet of white roses, with the roses in various stages of bloom. The petals of the roses are delicate and pristine white, contrasting with the soft pink hues visible in their centers. The arrangement is full and lush, with stems protruding outwards. Throughout the video, there are no significant changes in the composition or positioning of the roses, and the background remains consistently blurred, ensuring the floral arrangement remains the focal point.</details>                                                              |                                      <details><summary>Caption</summary>The video showcases a majestic snow-capped mountain range against a cloudy sky, with the peaks covered in pristine white snow and jagged rocky outcrops protruding from the slopes. The mountains cast long shadows across the snow-covered terrain below. Initially, the sky is a vivid blue with wispy white clouds, but as the video progresses, the clouds become slightly more dispersed, revealing more of the blue sky. Throughout the video, the overall composition and grandeur of the mountain vistas remain consistent, maintaining the serene and awe-inspiring natural beauty of the landscape.</details>                                      |

#### 30B Model Stage 2

|                                                                                                                                                                                                                                                                                                                              256x256x455                                                                                                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                  256x256x455                                                                                                                                                                                                                                  |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                                                                                                                                                                                                                    <video src="https://github.com/user-attachments/assets/e5558081-8710-4474-a522-a19a573a22e4" />                                                                                                                                                                                                                                                                                    |                                                                                                                                                                                        <video src="https://github.com/user-attachments/assets/d4625360-75f4-489a-893d-e4341b644be1" />                                                                                                                                                                                        |
| <details><summary>Caption</summary>The video showcases a serene aerial view of a mountainous landscape, consistently blanketed in snow and clouds throughout its duration. The foreground prominently features rugged, snow-capped peaks with jagged rock formations piercing through the pristine white snow. The background is consistently filled with a vast expanse of billowing clouds, interspersed with patches of blue sky above. The overall scene maintains a sense of tranquility and natural beauty, highlighting the grandeur of the mountainous terrain without any noticeable changes in the composition or perspective of the aerial view.</details> | <details><summary>Caption</summary>The video begins with a serene winter landscape featuring a frozen body of water in the foreground. The ice-covered surface is smooth and reflective, with patches of exposed water visible. In the background, a dense forest of evergreen trees lines the far shore, their branches covered in snow. The scene is hazy, with a grayish tint suggesting overcast or foggy conditions, maintaining a wintry ambiance throughout.</details> |  

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] The Movie Gen team @ Meta. Movie Gen: A Cast of Media Foundation Models. 2024
