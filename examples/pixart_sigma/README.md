# PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation

## Introduction

PixArt-\Sigma is a Diffusion Transformer model~(DiT) capable of directly generating images at 4K resolution. PixArt-\Sigma represents a significant advancement over its predecessor, PixArt-\alpha, offering images of markedly higher fidelity and improved alignment with text prompts. A key feature of PixArt-\Sigma is its training efficiency. Leveraging the foundational pre-training of PixArt-\alpha, it evolves from the 'weaker' baseline to a 'stronger' model via incorporating higher quality data, a process called "weak-to-strong training". The advancements in PixArt-\Sigma are twofold: (1) High-Quality Training Data: PixArt-\Sigma incorporates superior-quality image data, paired with more precise and detailed image captions. (2) Efficient Token Compression: a novel attention module within the DiT framework is proposed that compresses both keys and values, significantly improving efficiency and facilitating ultra-high-resolution image generation. Thanks to these improvements, PixArt-\Sigma achieves superior image quality and user prompt adherence capabilities with significantly smaller model size (0.6B parameters) than existing text-to-image diffusion models, such as SDXL (2.6B parameters) and SD Cascade (5.1B parameters). Moreover, PixArt-\Sigma's capability to generate 4K images supports the creation of high-resolution posters and wallpapers, efficiently bolstering the production of high-quality visual content in industries such as film and gaming.

## Requirement

- Python >= 3.8
- Mindspore >= 2.3

## Getting Start

### Downloading Pretrained Checkpoints

We refer to the [official repository of PixArt-sigma](https://github.com/PixArt-alpha/PixArt-sigma) for pretrained checkpoints downloading.

After downloading the `PixArt-Sigma-XL-2-256x256.pth` and `PixArt-Sigma-XL-2-{}-MS.pth`, please place it under the `models/` directory, and then run `tools/convert.py` for each checkpoint separately. For example, to convert `models/PixArt-Sigma-XL-2-1024-MS.pth`, you can run:

```bash
python tools/convert.py --source models/PixArt-Sigma-XL-2-1024-MS.pth --target models/PixArt-Sigma-XL-2-1024-MS.ckpt
```

In addition, please download the [VAE checkpoint](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae), [T5 checkpoint](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/text_encoder), [T5 Tokenizer](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/tokenizer) and put them under `models` directory.


After conversion, the checkpoints under `models/` should be like:
```bash
models/
├── PixArt-Sigma-XL-2-256x256.ckpt
├── PixArt-Sigma-XL-2-512-MS.ckpt
├── PixArt-Sigma-XL-2-1024-MS.ckpt
├── PixArt-Sigma-XL-2-2K-MS.ckpt
├── vae/
├── tokenizer/
└── text_encoder/
```

### Sampling using Pretrained model

You can then run the sampling using `sample.py`. For examples, to sample a 512x512 resolution image, you may run

```bash
python sample.py -c configs/inference/pixart-sigma-512-MS.yaml --prompt "your magic prompt"
```

For higher resolution images, you can choose either `configs/inference/pixart-sigma-1024-MS.yaml` or `configs/inference/pixart-sigma-2K-MS.yaml`.

And to sample an image with a varying aspect ratio, you need to use the flag `--image_width` and `--image_width`. For example, to sample a 512x1024 image, you may run

```bash
python sample.py -c configs/inference/pixart-sigma-1024-MS.yaml --prompt "your magic prompt" --image_width 1024 --image_height 512
```

The following demo image is generated using the default prompt with the command:

```bash
python sample.py -c configs/inference/pixart-sigma-1024-MS.yaml --image_width 1024 --image_height 512 --seed 1024
```
<p align="center"><img width="1024" src="https://github.com/user-attachments/assets/d2a4a391-744e-4ae8-a035-427a26e2c655"/>