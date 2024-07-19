# Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis (PixArt-Alpha)

## Introduction

The most advanced text-to-image (T2I) models require significant training costs, seriously hindering the fundamental innovation for the AIGC community while increasing CO2 emissions. This paper introduces PIXART-α, a Transformer-based T2I diffusion model whose image generation quality is competitive with state-of-the-art image generators (e.g., Imagen, SDXL, and even Midjourney), reaching near-commercial application standards. Additionally, it supports high-resolution image synthesis up to 1024px resolution with low training cost. To achieve this goal, three core designs are proposed: (1) Training strategy decomposition: The authors devise three distinct training steps that separately optimize pixel dependency, text-image alignment, and image aesthetic quality; (2) Efficient T2I Transformer: the cross-attention modules is incorporate into Diffusion Transformer (DiT) to inject text conditions and streamline the computation-intensive class-condition branch; (3) High-informative data: the authors emphasize the significance of concept density in text-image pairs and leverage a large Vision-Language model to auto-label dense pseudo-captions to assist text-image alignment learning. As a result, PIXART-α's training speed markedly surpasses existing large-scale T2I models, e.g., PIXART-α only takes 10.8% of Stable Diffusion v1.5's training time (675 vs. 6,250 A100 GPU days), saving nearly $300,000 ($26,000 vs. $320,000) and reducing 90% CO2 emissions. Moreover, compared with a larger SOTA model, RAPHAEL, the training cost is merely 1%. Extensive experiments demonstrate that PIXART-α excels in image quality, artistry, and semantic control. PIXART-α provides new insights to the AIGC community and startups to accelerate building their own high-quality yet low-cost generative models from scratch.

## Requirement

- Mindspore >= 2.3

## Getting Start

### Downloading Pretrained Checkpoints

We refer to the [official repository of PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) for pretrained checkpoints downloading.

After downloading the `PixArt-XL-2-{}x{}.pth` file and `PixArt-XL-2-1024-MS.pth`, please place it under the `models/` directory, and then run `tools/convert.py`. For example, to convert `models/PixArt-XL-2-1024-MS.pth`, you can run:

```bash
python tools/convert.py --source models/PixArt-XL-2-1024-MS.pth --target models/PixArt-XL-2-1024-MS.ckpt
```

In addition, please download the [VAE checkpoint](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema) and [T5 checkpoint](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) and put them under `models` directory, and convert the checkpoints by running:

```bash
python tools/convert_vae.py --source models/sd-vae-ft-ema/diffusion_pytorch_model.bin --target models/sd-vae-ft-ema.ckpt
```

```bash
python tools/convert_t5.py --source models/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin models/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin --target models/t5-v1_1-xxl/model.ckpt
```

After conversion, the checkpoints under `models/` should be like:
```bash
models/
├── PixArt-XL-2-256x256.ckpt
├── PixArt-XL-2-512x512.ckpt
├── PixArt-XL-2-1024-MS.ckpt
├── sd-vae-ft-ema.ckpt
└── t5-v1_1-xxl/model.ckpt
```

### Sampling using Pretrained model

You can then run the sampling using `sample.py`. For examples, to sample a 512x512 image, you may run

```bash
python sample.py -c configs/inference/pixart-512x512.yaml --prompt "your magic prompt"
```

And to sample a image with varying aspect ratio, you may use the flag `--image_width` and `--image_width` together with the configuration file `configs/inference/pixart-1024-MS.yaml` to alter the output size. For example, to sample a 512x1024 image, you may run

```bash
python sample.py -c configs/inference/pixart-1024-MS.yaml --prompt "your magic prompt" --image_width 1024 --image_height 512
```

The following demo image is generated using the default prompt with the command:

```bash
python sample.py -c configs/inference/pixart-1024-MS.yaml --image_width 1024 --image_height 512 --seed 0
```
<p align="center"><img width="1024" src="https://github.com/zhtmike/mindone/assets/8342575/741e7a0a-11ab-4377-a8cd-77e689353c1f"/>


### Finetuning

We support finetune PixArt-Alpha model for 512x512 resolution on 910* Ascend device. Finetuning PixArt-Alpha with multi scale resolution is still in progress. To start with, please prepare the dataset and put it in the following format:

```text
data_path
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

where `img_txt.csv` is the image-caption file annotated in the following format. i.e.,

```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

Then, please extract the T5 embedding and VAE caching using the following command.

```bash
python infer_t5.py --csv_path path_to_the_csv_file --output_path path_of_the_output_directory
```

```bash
python infer_vae.py --csv_path path_to_the_csv_file --image_dir path_storing_image_files --output_path path_of_the_output_directory
```

After that, you can start the training by run

```bash
python train.py -c configs/train/pixart-512x512.yaml \
    --csv_path path_to_the_csv_file \
    --latent_dir path_storing_the_VAE_latent \
    --text_emb_dir path_storing_the_T5_embedding
```

And the result is saved at `./output` directory.

Or you can start a distributed training by running

```bash
msrun --worker_num=8 --local_worker_num=8 --log_dir="./logs"  \
    train.py --config configs/train/pixart-512x512.yaml \
    --csv_path path_to_the_csv_file \
    --latent_dir path_storing_the_VAE_latent \
    --text_emb_dir path_storing_the_T5_embedding
    --use_parallel True
```

For detail usage of the training script, you can run `python train.py -h`.


## Benchmark

### Training

| Model       | Context       | Global Batch Size x Grad. Accu. | Max. Resolution | Acceleration | FPS (img/s) |
|-------------|---------------|---------------------------------|-----------------|--------------|-------------|
| PixArt-XL-2 | D910*x4-MS2.3 | 64x1                            | 512x512         | FP16, FA, O1 | 18.55       |

> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
> Acceleration: FP16: float16 computation; FA: Flash Attention enabled; O1: Refer to Mindspore `jit_level` document.
> Max. Resolution: The maximum resolution of the image in training.
> FPS: images per second during training. average training time (s/step) = batch_size / FPS
