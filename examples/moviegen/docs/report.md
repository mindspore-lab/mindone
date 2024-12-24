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

[//]: # (TODO: Figure 3 Overall architecture)

### TAE

For improved training and inference efficiency, we perform generation in a spatio-temporally compressed latent space.

### Transformer Backbone

[//]: # (TODO: Figure 8)

Movie Gen uses the [LLaMa3](https://arxiv.org/abs/2407.21783) backbone architecture for the joint image-video generation
model, enabling confident scaling of the model size while maintaining efficient training. It can directly generate video
at different aspect ratios (e.g., 1:1, 9:16, 16:9) and multiple lengths (4 – 16 seconds) at 768 px resolution.

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

Different parallelization strategies are depicted in the Transformer block figure.

[//]: # (TODO: add reference to the figure.)

- **Tensor-parallelism (TP)** shards the weights of linear layers either along columns or rows, and results in each NPU
  involved in the sharding performing _tp-size_ less work (FLOPs) and generating _tp-size_ fewer activations for
  column-parallel shards and consuming _tp-size_ fewer activations for row-parallel shards. The cost of performing such
  a sharding is the addition of all-reduce communication overheads in both the forward (row-parallel) and backward
  (column-parallel) passes.
- **Sequence-parallelism (SP)** builds upon TP to also allow the sharding of the input over the sequence dimension for
  layers which are replicated and in which each sequence element can be treated independently. Such layers, e.g.,
  LayerNorm, would otherwise perform duplicate compute and generate identical (and thus replicated) activations across
  the TP-group.
- **Context-parallelism (CP)** enables a partial sharding over the sequence dimension for the _sequence-dependent
  softmax-attention operation_. CP leverages the insight that for any given (_source_ (_context_), _target_ (_query_))
  sequences pair, _softmax-attention is only sequence-dependent over the context and not the query_. Therefore, in the
  case of self-attention where the input source and target sequences are identical, CP allows the attention computation
  to be performed with only an all-gather for the $K$ and $V$ projections (instead of $Q$, $K$, and $V$) in the forward
  pass, and a reduce-scatter for their associated gradients in the backward.
- **Fully sharded data parallel (FSDP)** shards the model, optimizer, and gradients across all data-parallel NPUs,
  synchronously gathering and scattering parameters and gradients throughout each training step.

### Text Encoders

Movie Gen uses a combination of [UL2](https://arxiv.org/abs/2205.05131), [ByT5](https://arxiv.org/abs/2105.13626), and
Long-prompt [MetaCLIP](https://arxiv.org/abs/2309.16671) as text encoders to provide both semantic-level and
character-level text understanding for the backbone:

- **UL2** is trained using massive text-only data and potentially provides strong text reasoning abilities in its
  features.
- **Long-prompt MetaCLIP** provides text representations that are aligned with visual representations that are
  beneficial for cross-modal generation.
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

## Inference

Movie Gen uses a simple first-order Euler ODE solver with a unique t-schedule tailored to the model. Specifically, the
quality of an N-step video generation process can be closely approximated with merely 50 steps by implementing a
**linear-quadratic t-schedule**.

[//]: # (TODO: Add figure 10 visualization)

This approach follows the first 25 steps of an $N$-step linear schedule and then approximates the remaining $N-25$ steps
with 25 quadratically placed steps. The linear-quadratic strategy is predicated on the observation that the first
inference steps are pivotal in setting up the scene and motion of the video, since most changes occur in the first
solver steps (the left figure above).

[//]: # (TODO: replace (the left figure above\) with the fig number)
