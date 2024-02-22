# Scalable Diffusion Models with Transformers (DiT)

## Introduction


## Get Started

### Environment Setup

### Pretrained Checkpoints

We refer to the [official repository of DiT](https://github.com/facebookresearch/DiT) for pretrained checkpoints downloading. Currently, only two checkpoints `DiT-XL-2-256x256` and `DiT-XL-2-512x512` are available.

After downloading the `DiT-XL-2-{}x{}.pt` file, please place it under the `models/` folder, and then run `tools/dit_converter.py`. For example, to convert `models/DiT-XL-2-256x256.pt`, you can run:
```bash
python tools/dit_converter.py --source models/DiT-XL-2-256x256.pt --target models/DiT-XL-2-256x256.ckpt
```

In addition, please download the VAE checkpoint from [huggingface/stabilityai.co](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main), and convert this VAE checkpoint by running:
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-mse.ckpt
```

After conversion, the checkpoints under `models/` should be like:
```bash
models/
├── DiT-XL-2-256x256.ckpt
├── DiT-XL-2-512x512.ckpt
└── sd-vae-ft-mse.ckpt
```

## Sampling

To run inference of `DiT-XL/2` model with the `512x512` image size on Ascend devices, you can use:
```bash
python sample.py --image-size 512 --dit_checkpoint models/DiT-XL-2-512x512.ckpt --seed 42
```
To run inference of `DiT-XL/2` model with the `256x256` image size on Ascend devices, you can use:
```bash
python sample.py --image-size 256 --dit_checkpoint models/DiT-XL-2-256x256.ckpt --seed 42
```
To run the same inference on GPU devices, simply set `--device_target GPU` for the commands above.

You can also adjust the classifier-free guidance scale by setting `--guidance_scale`. The default guidance scale is $8.5$.


## Training
