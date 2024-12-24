# Movie Gen

Movie Gen is a family of foundation models that can natively generate high-fidelity images and videos
while also possessing the abilities to edit and personalize the videos.

Meta researchers found that scaling the training data, compute, and model parameters of a simple
Transformer-based ([LLaMa3](https://arxiv.org/abs/2407.21783)) model trained with
[Flow Matching](https://arxiv.org/abs/2210.02747) yields high-quality generative models for video or audio.

Movie Gen supports the following features:

1. Text-to-Video synthesis
2. Video personalization
3. Video editing

# Detailed Technical Report

## Architecture Overview

<p align="center">
<img alt="Architecture Overview" width="750" src="https://github.com/user-attachments/assets/c58c3c13-d300-4c9c-9737-f968cca3618f"/>
<br><em>Figure 1. Overall architecture of Movie Gen</em>
</p>

### TAE

For improved training and inference efficiency, we perform generation in a spatio-temporally compressed latent space.

### Transformer Backbone

<p align="center">
<img alt="Transformer backbone and model parallelism" width="700" src="https://github.com/user-attachments/assets/87811e4f-5e49-4530-b43f-734782bf0e0a"/>
<br><em>Figure 2. Transformer backbone and model parallelism</em>
</p>

Movie Gen uses the [LLaMa3](https://arxiv.org/abs/2407.21783) backbone architecture for the joint image-video generation
model, enabling confident scaling of the model size while maintaining efficient training. It can directly generate video
at different aspect ratios (e.g., 1:1, 9:16, 16:9) and multiple lengths (4 – 16 seconds) at 768 px resolution.

Links: [LLaMa3 Backbone](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/network.py#L273),
[Transformer Block](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/network.py#L52).

There are three changes to the LLaMa3 Transformer block for the use case of video generation using Flow Matching:

1. Add a cross-attention module between the self-attention module and the feed forward network (FFN)
   to each Transformer block to incorporate text conditioning based on the text prompt embedding **P**.
   Multiple different text encoders are leveraged due to their complementary strengths
   (see [Text Encoders](#text-encoders)).
2. Add adaptive layer norm blocks to incorporate the time-step $t$ to the Transformer, as used in prior work
   ([DiT](https://arxiv.org/abs/2212.09748)).
3. Use full bidirectional attention instead of causal attention used in language modeling.

#### Differences Among Models

The Movie Gen family of models contains the following variations: 1B, 5B, and 30B parameters.

| Model | Layers | Model Dimension | FFN Dimension | Attention Heads |
|:-----:|:------:|:---------------:|:-------------:|:---------------:|
|  1B   |   24   |      1536       |     4096      |       16        |
|  5B   |   32   |      3072       |     8192      |       24        |
|  30B  |   48   |      6144       |     16384     |       48        |

#### Patchifying Inputs

To prepare inputs for the Transformer backbone, the video latent code ($T \times C \times H \times W$) is first
'patchified' using a 3D convolutional layer (as in [here](https://arxiv.org/abs/2010.11929)) and then flattened to yield
a 1D sequence. The 3D convolutional layer uses a kernel size of $k_t \times k_h \times k_w$ with a stride equal to the
kernel size and projects it into the same dimensions as needed by the Transformer backbone. Thus, the total number of
input tokens to the Transformer backbone is $THW/(k_tk_hk_w)$. We use $k_t=1$ and $k_h=k_w=2$, i.e., we produce
$2 \times 2$ spatial patches.

#### Learnable Positional Embedding (PE)

Movie Gen uses a factorized learnable positional embedding to enable arbitrary size, aspect ratio, and video
length (as in [NaViT](https://arxiv.org/abs/2307.06304)) inputs to the Transformer.
The 'patchified' tokens, i.e., output of the 3D convolutional layer, are converted into separate embeddings $\phi_h$,
$\phi_w$ and $\phi_t$ of spatial $h$, $w$, and temporal $t$ coordinates. The final positional embeddings are calculated
by adding all the factorized positional embeddings together. Finally, the final positional embeddings **are added
to the input for all the Transformer layers**. Compared with adding the positional embeddings to the first layer only,
adding to all layers can effectively reduce the distortion and morphing artifacts, especially in the temporal dimension.

#### Model Parallelism

Movie Gen employs 3D parallelism to support model-level scaling across three axes: number of parameters, input tokens,
and dataset size, while also allowing horizontal scale-out to more NPUs. It utilizes a combination of [fully sharded
data parallelism](https://arxiv.org/abs/2304.11277), [tensor parallelism](https://arxiv.org/abs/1909.08053),
[sequence parallelism](https://arxiv.org/abs/2105.13120), and context parallelism.

Different parallelization strategies are depicted in the [Transformer block figure](#transformer-backbone).

[//]: # (TODO: fix the link above)

- **Tensor-parallelism (TP)**
  \[[TP](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/block.py#L59),
  [FusedTP](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/block.py#L91)]
  shards the weights of linear layers either along columns or rows, and results in each NPU involved in the sharding
  performing _tp-size_ less work (FLOPs) and generating _tp-size_ fewer activations for column-parallel shards and
  consuming _tp-size_ fewer activations for row-parallel shards. The cost of performing such a sharding is the addition
  of all-reduce communication overheads in both the forward (row-parallel) and backward (column-parallel) passes.
- **Sequence-parallelism (SP)**
  \[[code](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/network.py#L494)]
  builds upon TP to also allow the sharding of the input over the sequence dimension for layers which are replicated and
  in which each sequence element can be treated independently. Such layers, e.g., LayerNorm, would otherwise perform
  duplicate compute and generate identical (and thus replicated) activations across the TP-group.
- **Context-parallelism (CP)**
  \[[CP Attention](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/block.py#L210),
  [CP FlashAttention](https://github.com/hadipash/mindone/blob/5aa1e4dc91d71934905319ba984704d4d4a62f8b/examples/moviegen/mg/models/llama/block.py#L340)]
  enables a partial sharding over the sequence dimension for the _sequence-dependent softmax-attention operation_. CP
  leverages the insight that for any given (_source_ (_context_), _target_ (_query_)) sequences pair, _softmax-attention
  is only sequence-dependent over the context and not the query_. Therefore, in the case of self-attention where the
  input source and target sequences are identical, CP allows the attention computation to be performed with only an
  all-gather for the $K$ and $V$ projections (instead of $Q$, $K$, and $V$) in the forward pass, and a reduce-scatter
  for their associated gradients in the backward.
- **Fully sharded data parallel (FSDP)** shards the model, optimizer, and gradients across all data-parallel NPUs,
  synchronously gathering and scattering parameters and gradients throughout each training step.

### Text Encoders

Movie Gen uses a combination of [UL2](https://arxiv.org/abs/2205.05131), [ByT5](https://arxiv.org/abs/2105.13626), and
Long-prompt [MetaCLIP](https://arxiv.org/abs/2309.16671) as text encoders to provide both semantic-level and
character-level text understanding for the backbone:

- **UL2** is trained using massive text-only data and potentially provides strong text reasoning abilities in its
  features.
- **Long-prompt MetaCLIP** (Optional at this moment) provides text representations that are aligned with visual
  representations that are beneficial for cross-modal generation.
- **ByT5** encoder is only used to encode visual text, i.e., the part of the text prompt that explicitly asks for a
  character string to be generated in the output image / video.

The text embeddings from the three text encoders are concatenated after adding separate linear projection and LayerNorm
layers to project them into the same 6144 dimension space and normalize the embeddings.

## Training Details

Movie Gen is trained jointly on images and videos. Images are treated as single frame videos, enabling the use of the
same model to generate both images and videos. Compared to video data, paired image-text datasets are easier to scale
with diverse concepts and styles, and thus joint modeling of image and video leads to better generalization.

Training is performed in multiple stages for better efficiency:

1. Pre-raining on low-resolution 256 px images only.  
   Meta researchers observed that directly training T2I/V models from scratch results in a slower convergence speed than
   initializing them from a T2I model.
2. Joint training on low-resolution 256 px images and videos.  
   To enable the joint training, we double the spatial [PE](#learnable-positional-embedding-pe) layers to accommodate
   various aspect ratios, add new temporal PE layers to support up to 32 latent frames, and initialize spatial PE layers
   from the T2I model with 2x expansion.
3. Joint training at 768 px resolution.  
   For this stage, we expand the spatial PE layers by 3x.
4. Fine-tune the model on high-quality videos to improve the generations.  
   Improve the motion and aesthetic quality of the generated videos by fine-tuning the pre-trained model on a small
   fine-tuning set of manually selected videos. During this stage, multiple models are trained and combined to form the
   final model through a model averaging approach ([LLaMa3](https://arxiv.org/abs/2407.21783)).

### Training Objective

Movie Gen is trained with the [Flow Matching](https://arxiv.org/abs/2210.02747) framework,
i.e., it is trained to predict the velocity $V_t = \frac{dX_t}{dt}$ which teaches it to 'move' the sample $X_t$
in the direction of the video sample $X_1$.
Movie Gen uses simple linear interpolation or the optimal transport path (Lipman et al., 2023), i.e.,
$$X_t=tX_1+(1-(1-\sigma_{min})t)X_0$$
Where $\sigma_{min}=10^{-5}$. Thus, the ground truth velocity can be derived as:
$$V_t = \frac{dX_t}{dt} = X_1 - (1-\sigma_{min})X_0$$

The model parameters are denoted by $\theta$, the embedding of text prompts by **P**, and the predicted velocity
by $u(X_t, P, t)$.
The model is trained by minimizing the mean squared error between the ground truth velocity and model prediction:
$$E_{t,X_0,X_1,P}\|u(X_t, P, t;\theta)-V_t\|^2$$
As in prior work ([SD3](https://arxiv.org/abs/2403.03206)), $t$ is sampled from a logit-normal distribution where
the underlying Gaussian distribution has zero mean and unit standard deviation.

### Signal-to-Noise Ratio (SNR)

Choosing the right diffusion noise scheduler with a zero terminal signal-to-noise ratio is
[particularly important](https://arxiv.org/abs/2305.08891) for video generation.
Flow Matching implementation naturally ensures zero terminal SNR (i.e., at $t=0$).
This guarantees that, during training, the model receives pure Gaussian noise samples and is trained to predict the
velocity for them.
Thus, at inference, when the model receives pure Gaussian noise at $t = 0$, it can make a reasonable prediction.

### Bucketization for Variable Duration and Size

To accommodate diverse video lengths and aspect ratios, we bucketize the training data according to aspect ratio and
length. The videos in each bucket lead to the exact same latent shape which allows for easy batching of training data.

### Controlling FPS

The model is trained by pre-appending the sampling FPS value of each training video to the input text prompt
(e.g., “FPS-16”).

### Validation During Training

Meta researchers observed that the validation loss is well correlated with human evaluation results as the later
checkpoints with lower validation loss perform better in the human evaluations. This suggests that the Flow Matching
validation loss can serve as a useful proxy for evaluations during model development. Similar observation was made by
the authors of [SD3](https://arxiv.org/abs/2403.03206). For this reason, we maintain a validation set of unseen videos
and monitor the validation loss throughout training.

### Learning Rate Reduction

We decrease the learning rate by half whenever the validation loss plateaus to continue improving the model performance.

### Personalization

Enables the video generation model to condition on a text as well as an image of a person to generate a video featuring
the chosen person.
The generated personalized video maintains the identity of the person while following the text prompt.

### Editing

It allows users to effortlessly perform precise and imaginative edits on both real and generated videos using a textual
instruction. Since large-scale supervised video editing data is harder to obtain, the researchers show a novel approach
to training such a video editing model without supervised video editing data.

### Performance

Experiments were conducted on Ascend 910* using MindSpore 2.3.1 in Graph mode.

> [!NOTE]
> We trained all the models using BF16 precision.

| Model | Cards |   Stage   |       Batch size        |       Resolution        |        Recompute        | TAE Cache | Time (s/step) |                              Config                               |
|:-----:|:-----:|:---------:|:-----------------------:|:-----------------------:|:-----------------------:|:---------:|:-------------:|:-----------------------------------------------------------------:|
|  30B  |   8   |  2 (T2V)  |        Video: 1         |       256x256x455       |           ON            |    Yes    |     23.8      | [stage2_t2iv_256px.yaml](../configs/train/stage2_t2iv_256px.yaml) |
|  5B   |   8   |  1 (T2I)  |           10            |         256x455         |           ON            |    Yes    |     1.29      |  [stage1_t2i_256px.yaml](../configs/train/stage1_t2i_256px.yaml)  |
|  5B   |   8   | 2 (T2I/V) |  Image: 1<br/>Video: 1  | 256x455<br/>256 frames  | ON<br/>(Every 2 blocks) |    Yes    |     5.09      | [stage2_t2iv_256px.yaml](../configs/train/stage2_t2iv_256px.yaml) |
|  5B   |   8   | 3 (T2I/V) |  Image: 1<br/>Video: 1  | 576x1024<br/>256 frames |           ON            |    Yes    |     88.5      | [stage3_t2iv_768px.yaml](../configs/train/stage3_t2iv_768px.yaml) |
|  1B   |   8   |  1 (T2I)  |           10            |         256x455         |           ON            |    Yes    |     0.53      |  [stage1_t2i_256px.yaml](../configs/train/stage1_t2i_256px.yaml)  |
|  1B   |   8   | 2 (T2I/V) | Image: 10<br/>Video: 10 |  256x455<br/>32 frames  |           ON            |    Yes    |     2.07      | [stage2_t2iv_256px.yaml](../configs/train/stage2_t2iv_256px.yaml) |

### Execution Scripts

#### Stage 1

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
python train.py \
  --config configs/train/stage1_t2i_256px.yaml \
  --env.mode 0 \
  --env.jit_level O1 \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --train.settings.zero_stage 2 \
  --dataset.csv_path ../../datasets/mixkit-100videos/video_caption_train.csv \
  --dataset.video_folder ../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../datasets/mixkit-100videos/tae_latent_images \
  --dataset.text_emb_folder.ul2 ../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../datasets/mixkit-100videos/byt5_emb_100 \
  --dataloader.batch_size10 \
  --valid.dataset "" \
  --train.ema "" \
  --train.optimizer.weight_decay 0 \
  --train.save.ckpt_save_policy latest_k \
  --train.steps 2000 \
  --train.output_path "$stage1_dir"
```

</details>

#### Stage 2

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
python train.py \
  --config configs/train/stage2_t2iv_256px.yaml \
  --env.mode 0 \
  --env.jit_level O1 \
  --env.max_device_memory 59GB \
  --env.distributed True \
  --model.pretrained_model_path "$stage1_dir"/ckpt/llama-5B-s2000.ckpt\
  --train.settings.zero_stage 2 \
  --dataset.csv_path ../../datasets/mixkit-100videos/video_caption_train.csv \
  --dataset.video_folder ../../datasets/mixkit-100videos/mixkit \
  --dataset.tae_latent_folder ../../datasets/mixkit-100videos/tae_latent \
  --dataset.text_emb_folder.ul2 ../../datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 ../../datasets/mixkit-100videos/byt5_emb_100 \
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

#### Stage 3

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
python train.py \
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
  --dataloader.batch_size.image_batch_size 1 \
  --dataloader.batch_size.video_batch_size 1 \
  --train.ema "" \
  --train.optimizer.weight_decay 0 \
  --train.settings.gradient_accumulation_steps 5 \
  --train.steps 30000 \
  --train.output_path "$stage3_dir"
```

</details>

## Inference

Movie Gen uses a simple first-order Euler ODE solver with a unique t-schedule tailored to the model. Specifically, the
quality of an N-step video generation process can be closely approximated with merely 50 steps by implementing a
**linear-quadratic t-schedule**.

<p align="center">
<img alt="The linear-quadratic t-schedule" height="250" src="https://github.com/user-attachments/assets/888080ac-b162-4de0-a420-b2fb00c66fff"/>
<br><em>Figure 3. The linear-quadratic t-schedule</em>
</p>

This approach follows the first 25 steps of an $N$-step linear schedule and then approximates the remaining $N-25$ steps
with 25 quadratically placed steps. The linear-quadratic strategy is predicated on the observation that the first
inference steps are pivotal in setting up the scene and motion of the video, since most changes occur in the first
solver steps ([Figure 3a](#inference)).

[//]: # (TODO: fix the link above)

### Generated Video Examples

|                                                                                                                                                                                                                                                                                                                                         256x256x455                                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                                                                                                             256x256x455                                                                                                                                                                                                                                                                                                                                                              |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                                                                                                                                                                                                                               <video src="https://github.com/user-attachments/assets/84264678-a2c4-4605-93c7-4efce8b4647a" />                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                   <video src="https://github.com/user-attachments/assets/357c93f1-b129-4441-80dc-adbe0d089a3a" />                                                                                                                                                                                                                                                                                                                    |
| <details><summary>Caption</summary>The video showcases a person wearing a blue cap and a plaid shirt, sitting on the ground with a golden retriever dog. The person is seen engaging in an affectionate interaction with the dog, gently stroking its fur and at one point, caressing or scratching behind the dog's ears. Throughout the video, the dog remains relaxed and content, with its mouth slightly open as if panting or smiling. The setting is an outdoor grassy area with fallen leaves or twigs scattered on the ground, under warm lighting that creates a cozy, intimate atmosphere focused on the bonding moment between the person and their canine companion.</details> | <details><summary>Caption</summary>The video features a close-up view of a cat with striking blue eyes and a white furry face adorned with brown and black stripes on its head. Initially, the cat is seen looking directly at the camera with an attentive expression, held gently by a human hand around its neck area against a blurred indoor background with a brown surface. As the video progresses, the cat's gaze becomes more intense and focused, with its whiskers appearing more prominent and alert. The camera zooms in slightly, cropping out some of the surrounding area to bring the cat's face into closer view, maintaining the attentive and engaged demeanor of the feline throughout the sequence.</details> |
|                                                                                                                                                                                                                                                                                               <video src="https://github.com/user-attachments/assets/e89a6be6-1e5b-4508-8980-89d824824e34" />                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                   <video src="https://github.com/user-attachments/assets/70cdc452-cad8-4781-9975-1c9feb8b89d6" />                                                                                                                                                                                                                                                                                                                    |
|                                                              <details><summary>Caption</summary>The video showcases a static image of a bouquet of white roses, with the roses in various stages of bloom. The petals of the roses are delicate and pristine white, contrasting with the soft pink hues visible in their centers. The arrangement is full and lush, with stems protruding outwards. Throughout the video, there are no significant changes in the composition or positioning of the roses, and the background remains consistently blurred, ensuring the floral arrangement remains the focal point.</details>                                                              |                                      <details><summary>Caption</summary>The video showcases a majestic snow-capped mountain range against a cloudy sky, with the peaks covered in pristine white snow and jagged rocky outcrops protruding from the slopes. The mountains cast long shadows across the snow-covered terrain below. Initially, the sky is a vivid blue with wispy white clouds, but as the video progresses, the clouds become slightly more dispersed, revealing more of the blue sky. Throughout the video, the overall composition and grandeur of the mountain vistas remain consistent, maintaining the serene and awe-inspiring natural beauty of the landscape.</details>                                      |
