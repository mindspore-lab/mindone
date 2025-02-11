# Awesome Large Vision Models and Foundation Models

This page keeps tracking awesome large vision models (LVM) and their dependent foundation models (FM).

We also analyze and show their training or pretraining methods and datasets for better insight.

Under construction and update.

## Contents

- [Large Vision Models](#large-vision-models)
  - [Image Generation](#image-generation)
  - [Segmentation](#segmentation)
- [Foundation Models](#foundation-models)
  - [Vision](#vision)
  - [Language](#language)
  - [Vision-Language Representation](#vision-language-representation)
- [Abbreviation Explain](#abbreviation-explain)


## Large Vision Models

### Image Generation

> [Stable Diffusion 2.1](): *High-Resolution Image Synthesis with Latent Diffusion Models. [[Stability AI]]() Dec 2022 [[open]](https://github.com/Stability-AI/StableDiffusion)*
>> Rely on: [OpenCLIP]()
  ```yaml
  Field: Image Generation (text-to-image w/ extendable conditions)
  Params: ~1B
  Training Data: LAION-5B (5B images) with NSFW filtering
  Training Algo: Diffusion DDPM
  Arch: LDM
      - AutoEncoder-KL: (Frozen)
        Params: 84M
        Pretrain Data: OpenImages (~9M images)
        Pretrain Algo: Generative Adversarial Training
        Core Operator: ResBlock, Transformer
      - OpenCLIP-TextEncoder: (ViT-H/14 version, Frozen)
        Pretrain Data: LAION-2B
        Pratrain Algo: Contrastive Learning
        Archicture: BERT-like Transformer
      - UNet for Diffusion
        Params: 865M
        Core Operator: ResBlock, Cross-attention Transformer
  Resolution: 768x768
  ```

> [Stable Diffusion 1.x](): *High-Resolution Image Synthesis with Latent Diffusion Models. [[Stability AI]]() Aug 2022 [[open]](https://github.com/Stability-AI/StableDiffusion)*
>> Rely on: [CLIP]()

  ```yaml
  Main Diff from SD 2.1:
   Training Data: LAION-400M
   CLIP-TextEncoder: (ViT-L/14 version)
     Pretrain Data: YFCC-15M / WIT-400M
   Resolution: 256x256
  ```

> [ControlNet](): *Adding Conditional Control to Text-to-Image Diffusion Models. Feb 2023 [[open]]()*
>> Rely on: [OpenCLIP]()

  ```yaml
  Field: Image Generation (image+text -> image)
  Params: ~1.5B
  Training Data: 3M+ edge-image-caption pairs, 600k edge-image-caption pairs, 80k pose-image-caption pairs, other depth/segmentation/sketch/noraml-image-caption pairs
  Training Algo: Diffusion DDPM, freeze SD and train SD Encoder + Zero Convolution
  Archicture: Stable Diffusion, SD Encoder + Zero Convolution
  ```

> [Imagen](): *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. [Google]() May 2022*
>> Rely on: [T5 model]()

  ```yaml
  Field: Image Generation (text to image)
  Params: 7.6B
  Training Data: internal datasets(≈ 460M image-text pairs), LAION-400M dataset
  Training Algo: Casacded Diffusion
  Archicture:
    - Frozen Text Encoder: T5-large
    - Text-to-Image Diffusion Model:
        Params: 2B
        Arch: UNet for Diffusion
    - Upsampler1: 64x64 -> 256x256
        Params: 600M
        Arch: Efficient UNet for Diffusion
    - Upsampler2: 256x256 -> 1024x1024
        Params: 400M
        Arch: Efficient UNet for Diffusion
  ```

> [Dall-e 2](): *Hierarchical Text-Conditional Image Generation with CLIP Latents. [OpenAI]() April 2022*
>> Rely on: [CLIP]()

  ```yaml
  Field: Image Generation
  Params: 6.5B
  Training Data: DALL-E dataset (~250M images)
  Training Algo: Diffusion
  Archicture:
    - CLIP-Encoder:
      ImageEncoder: ViT-H/16 (input size 256x256)
      TextEncoder:  BERT-like Transformer
      Pretrain Data: YFCC dataset and DALL-E datasets filtered for aesthetic quality and safety.  (~650M images)
    - Prior: a decoder-only Transformer
    - Upsampler1: 64x64 -> 256x256
        Params: 600M
        Arch: Efficient UNet for Diffusion
    - Upsampler2: 256x256 -> 1024x1024
        Params: 400M
        Arch: Efficient UNet for Diffusion
  ```

### Segmentation

> [SAM](): *Segment Anything. [Meta]() April 2023*
>> Rely on: [CLIP](), [MAE-ViT-H/16]()

  ```yaml
  Field: Image Segmentation
  Params: ~1B
  Training Data: SA-1B, containing 1B masks for 11M images, auto-annotation  (Core Contribution)
  Training Algo: Supervised, Focal Loss, Dice Loss
  Archicture:
    - ImageEncoder:
      Params: 632M
      Arch: ViT-H/16
      Pretrain Algo: MAE
    - PromptEncoder:
      Arch: CLIP TextEncoder
      Pretrain Algo: Contrastive Learning
    - MaskDecoder:
      Core Operator: Cross-attention Transformer
  ```

## Foundation Models

### Vision

> [DINO v2](): *DINOv2: Learning Robust Visual Features without Supervision. [Meta]() April 2023*

  ```yaml
  Arch: ViT
  Params: 1B
  Pretrain Data: LVD-142M (including imagenet-22k, ADE20K, and filtered internet images)
  Pretrain Algo: DINOv2 (~DINO + iBOT), CL, Teacher-student Discrimination,
  ```

> [DINO](): *Emerging Properties in Self-Supervised Vision Transformers. [Meta]() May 2021*

  ```yaml
  Arch: ViT
  Params: 1B
  Pretrain Data:
  Pretrain Algo: DINO, SL, CL, Teacher-student Discrimination
  ```

> [iBOT](): *iBOT: Image BERT Pre-Training with Online Tokenizer. [ByteDance]() Nov 2021*

> [MAE](): *Masked Autoencoders Are Scalable Vision Learners. [Meta]() Nov 2021*

  ```yaml
  Arch: Swinv2-H, ViT
  ```
> [SimMIM](): *SimMIM: A Simple Framework for Masked Image Modeling. [Microsoft]() Nov 2021*

  ```yaml
  Arch: Swinv2-H, ViT
  ```

> [BeiT](): *BEiT: BERT Pre-Training of Image Transformers. [Microsoft]() June 2021*

> [ViT-22B](): *Scaling Vision Transformers to 22 Billion Parameters. [Google]() Feb. 2023*

> [InternImage](): *InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions. [[Shanghai AI Lab]](https://github.com/OpenGVLab/InternImage) Nov. 2022 [[open]](https://github.com/OpenGVLab/InternImage)*

  ```yaml
  Field: Vision
  Params:
    InternImage-G: 3B
  Architecture: CNN
  Core Operator: Deformable Convolution v3
  ```

> [EVA-02]():

> [WSP-2B]():

> [MoCov2]():


#### Summary of Vision Foundation Models
<div align="left">

| Method       |  Objective  | Arch.   |  Params | Data    | Text-sup.  | LP Acc | Notes |
|:------------|:---------|:--------|:--------|:------|:------:|---------------|----------|
| MAE        |    MIM     | ViT-H/14 | 632M   |  IN-1k | &#10005;  | 83.3 |  |
| SimMIM     |    MIM     |   ViT-B       | 86M   |  IN-1k | &#10005;  | 83.8|  |
| SimMIM     |    MIM     |   Swinv2-H@224    | 658M   |  IN-1k | &#10005;  | 85.7|  |
| SimMIM     |    MIM     |   Swinv2-G@640    | 3B  |  IN-22K-ext | &#10005;  | 90.2| private dataset |
| BeiT       |    MIM     |    ViT-L/16   |  307M   |  IN-1k | &#10005;  | 85.2 |  |
| DINO       |    CL-Discrimination   | ViT-S/8    |   INet-1k   | IN-1k | &#10005;    |   85.5 | |
| iBOT       |    MIM     | ViT-L/16   |  307M    | IN-1k  |  &#10005;    | 81.0  |  |
| DINOv2     |    CL-Discrimination     | ViT-B/14   |  86M    |  LVD-142M  |  &#10005;    | 88.3 |   |
| DINOv2     |    CL-Discrimination     | ViT-L/14   |  307M    |  LVD-142M  |  &#10005;    | 89.5 |   |
| InternImage-XL | CL-M3I     |  CNN-DCNv3     |  335M    |  IN-22K  |  &#10005;    | 89.5 |   |
| InternImage-G |  CL-M3I     |  CNN-DCNv3     |  3B    |  Custom |  &#10005;    | 90.1 |   |
| Sup.       |    Cls    | ViT    |   22B   |  JFT-4B |  &#10005;    | 89.51 |   |

 </div>

*Note: For smaller vision backbones with good performance, please refer to [MindCV](https://github.com/mindspore-lab/mindcv/blob/main/benchmark_results.md)*

### Language

> [BERT]():

> [Roberta]():

> [T5 Model]():

### Vision-Language Representation

> [OpenCLIP](): *Reproducible Scaling Laws for Contrastive Language-Image Learning. Dec 2022*

```yaml
Objective: Constrastive Representation Learning (CRL)*
Dataset: YFCC-15M / WIT-400M
```

> [CLIP](): *Learning Transferable Visual Models From Natural Language Supervision. [OpenAI]() Feb 2021*

```yaml
Objective: Constrastive Representation Learning (CRL)
Dataset: LAION-2B
```

> [BeiT-3]():
*Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks. [Microsoft]() Aug 2022*

> [INTERN-2.5]():

#### Summary of Vision-Language Foundation Models
 <div align="left">

| Method       |  Objective  | ImgEnc | TxtEnc    |  Params | Data    | Text-sup.  | LP Acc | Notes |
|:------------|:------------|:--------|:--------|:--------|:---------|:------:|---------------|----------|
| CLIP        |    CL      | ViT-L/14 |  BERT-like   | 307M,    |  YFCC-15M | &#10003;  | 83.3 |  |
| OpenCLIP    |    CL      | ViT-H/14 |  BERT-like   | 632M,    |  LAION-2B | &#10003;  | 84.4 |  |

 </div>

## Abbreviation Explain

SL: Self-supervised Learning

Sup.: Supervised Learning

CL: Constrastive Learning, a family of SL methods

MIM: Masked Image Modeling, a faimly of SL methods
