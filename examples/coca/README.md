# CoCa: Contrastive Captioners are Image-Text Foundation Models

## Introduction

Coca is a a minimalist design to pretrain an image-text encoder-decoder foundation model jointly with contrastive loss and captioning loss, thereby subsuming model capabilities from contrastive approaches like CLIP and generative methods like SimVLM.
In contrast to standard encoder-decoder transformers where all decoder layers attend to encoder outputs, CoCa omits cross-attention in the first half of decoder layers to encode unimodal text representations, and cascades the remaining decoder layers which cross-attend to the image encoder for multimodal image-text representations.
The structure of DiT model and DiT blocks is shown below[<a href="#references">1</a>].

<p align="center">
  <img width="550" alt="figure" src="https://github.com/mlfoundations/open_clip/assets/52945530/ca75eccb-ea1f-4e9a-b628-1b88b691b849">
</p>
<p align="center">
  <em> Figure 1. Detailed illustration of CoCa architecture and training objectives. [<a href="#references">1</a>] </em>
</p>

## Get Started
In this tutorial, we will introduce how to run inference and finetuning experiments using MindONE.

### Environment Setup

```
pip install -r requirements.txt
```

### Pretrained Checkpoints

We provide a script for converting pre-trained weight from `.bin` to `.ckpt` in `tools/model_conversion/coca_convert.py`.

step1. Download the [Official](https://huggingface.co/laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/tree/main) pre-train weights `open_clip_pytorch_model.bin` from huggingface.

step2. Convert weight to MindSpore `.ckpt` format and put it to `./models/`.

After conversion, the checkpoints under `models/` should be like:
```bash
models/
├── coca_model.ckpt
```

## Inference

To run inference of Coca model to generate video caption, you can use:
```bash
python inference.py -c configs/coca_vit-l-14.yaml --video_path PATH_TO_YOUR_VIDEO --frame_type "middle"
```
By default, we use the middle frames of the video to generate caption. You can switch to use mutil_frames to generate caption by setting `--frame_type "mutil_frames"`.




# References

[1] Yu, Jiahui, et al. "Coca: Contrastive captioners are image-text foundation models."
