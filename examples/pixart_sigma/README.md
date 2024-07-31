# PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation

## Introduction


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
<p align="center"><img width="1024" src="https://github.com/zhtmike/mindone/assets/8342575/741e7a0a-11ab-4377-a8cd-77e689353c1f"/>
