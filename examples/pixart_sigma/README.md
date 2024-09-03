# PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation

## Introduction

PixArt-\Sigma is a Diffusion Transformer model~(DiT) capable of directly generating images at 4K resolution. PixArt-\Sigma represents a significant advancement over its predecessor, PixArt-\alpha, offering images of markedly higher fidelity and improved alignment with text prompts. A key feature of PixArt-\Sigma is its training efficiency. Leveraging the foundational pre-training of PixArt-\alpha, it evolves from the 'weaker' baseline to a 'stronger' model via incorporating higher quality data, a process called "weak-to-strong training". The advancements in PixArt-\Sigma are twofold: (1) High-Quality Training Data: PixArt-\Sigma incorporates superior-quality image data, paired with more precise and detailed image captions. (2) Efficient Token Compression: a novel attention module within the DiT framework is proposed that compresses both keys and values, significantly improving efficiency and facilitating ultra-high-resolution image generation. Thanks to these improvements, PixArt-\Sigma achieves superior image quality and user prompt adherence capabilities with significantly smaller model size (0.6B parameters) than existing text-to-image diffusion models, such as SDXL (2.6B parameters) and SD Cascade (5.1B parameters). Moreover, PixArt-\Sigma's capability to generate 4K images supports the creation of high-resolution posters and wallpapers, efficiently bolstering the production of high-quality visual content in industries such as film and gaming.

## Requirement

- CANN: 8.0
- Python >= 3.8
- Mindspore >= 2.3.1

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

### Vanilla Finetune

We support finetune PixArt-\Sigma model on 910* Ascend device.

#### Prepare the Dataset

- As an example, please download the `diffusiondb-pixelart` dataset from [this link](https://huggingface.co/datasets/jainr3/diffusiondb-pixelart). The dataset is a subset of the larger DiffusionDB 2M dataset, which has been transformed into pixel-style art.

- Once you have the dataset, create a label JSON file in the following format:
```json
[
    {
        "path": "file1.png",
        "prompt": "a beautiful photorealistic painting of cemetery urbex unfinished building building industrial architecture...",
        "sharegpt4v": "*caption from ShareGPT4V*",
        "height": 512,
        "width": 512,
        "ratio": 1.0,
    },
]
```
- Remember to
    - Replace `file1.png` with the actual image file path.
    - The `prompt` field contains a description of the image.
    - If you have captions generated from ShareGPT4V, add them to the `sharegpt4v` field. Otherwise, copy the label from the `prompt` line.
    - `height` and `width` field corresponds to the image height and width, and `ratio` corresponds to the value of `height` / `width`.

#### Finetune the Model:

Use the following command to start the finetuning process:

```bash
python train.py \
    -c configs/train/pixart-sigma-512-MS.yaml \
    --json_path path_to_your_label_file \
    --image_dir path_to_your_image_directory
```
- Remember to
    - Replace `path_to_your_label_file` with the actual path to your label JSON file.
    - Replace `path_to_your_image_directory` with the directory containing your images.

#### Distributed Training (Optional):

You can launch distributed training using multiple Ascend 910* Devices:

```bash
msrun --worker_num=8 --local_worker_num=8 --log_dir="log" train.py
    -c configs/train/pixart-sigma-512-MS.yaml \
    --json_path path_to_your_label_file \
    --image_dir path_to_your_image_directory \
    --use_parallel True
```
- Remember to
    - Replace `path_to_your_label_file` with the actual path to your label JSON file.
    - Replace `path_to_your_image_directory` with the directory containing your images.

## Benchmark

### Inference

| Context       | Scheduler | Steps | Resolution | Batch Size | Speed (step/s) | Config                                                                  |
|---------------|-----------|-------|------------|------------|----------------|-------------------------------------------------------------------------|
| D910*x1-MS2.3 | DPM++     | 20    | 256x256    | 1          | 18.04          | [pixart-sigma-256x256.yaml](configs/inference/pixart-sigma-256x256.yaml)|
| D910*x1-MS2.3 | DPM++     | 20    | 512x512    | 1          | 15.95          | [pixart-sigma-512-MS.yaml](configs/inference/pixart-sigma-512-MS.yaml)  |
| D910*x1-MS2.3 | DPM++     | 20    | 1024x1024  | 1          | 4.96           | [pixart-sigma-1024-MS.yaml](configs/inference/pixart-sigma-1024-MS.yaml)|
| D910*x1-MS2.3 | DPM++     | 20    | 2048x2048  | 1          | 0.57           | [pixart-sigma-2K-MS.yaml](configs/inference/pixart-sigma-2K-MS.yaml)    |

> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
> Speed (step/s): sampling speed measured in the number of sampling steps per second.

# References

[1] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, Zhenguo Li. PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation. arXiv:2403.04692, 2024.
