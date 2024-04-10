# AutoEncoders based on MindSpore

This repository contains SoTA image and video autoencoders and their training and inference pipelines implemented with MindSpore.

## Features
- VAE (Image Variational AutoEncoder)
    - [x] KL-reg with GAN loss (SD VAE)
    - [ ] VQ-reg with GAN loss (VQ-GAN)
- Causal 3D Autoencoder (Video AutoEncoder)
    - [ ] VQ-reg with GAN loss (MagViT)
    - [ ] KL-reg with GAN loss

## Installation

```
pip install -r requirements.txt
```

## Variational Autoencoder (VAE)

### Training

Please download the [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) checkpoint and put it under `models/`.

To train a VAE, please run
```
python train.py --config configs/training/your_train_receipe.yaml
```



For example, to train VAE-kl-f8 model on CelebA-HQ dataset, you can run
```
python train.py --config configs/training/vae_celeba.yaml
```
after setting the `data_path` argument to the dataset path.

Note that you can either set arguments by editing the yaml file, or parsing by CLI (e.g. appending `--data_path=datasets/celeba_hq/train` to the training command). The CLI arguments will overwrite the corresponding values in the base yaml config.

#### Key arguments:

- `model_config`: path to a yaml config file defining the autoencoder architecture and loss. Default: "configs/autoencoder_kl_f8.yaml"
- `use_discriminator`: If True, GAN adversarial training will be applied after `disc_start` steps (defined in model config). Default: False
- `device_target`: To run on GPUs, please set it to "GPU". Default: "Ascend"

<!--
Note that `calculate_adaptive_weight` is not used currently compared to torch GAN.
-->

For more arguments, please run `python train.py -h` to check.

### Evaluation

```
python infer.py \
    --model_config configs/autoencoder_kl_f8.yaml \
    --ckpt_path path/to/checkpoint \
    --data_path path/to/test_data \
    --size 256 \
```
After running it will save the reconstruction results in `samples/vae_recons` and report the PSNR and SSIM evaluation metrics by default.

For detailed arguments, please run `python infer.py -h`.

### Results on CelebA-HQ

We split the CelebA-HQ dataset into 24,000 images for training and 6,000 images for testing. After 22 epochs of training, the training performance and evaluation results on the test set are reported as follows.


| Model          |   Context   |  Precision         | Local BS x Grad. Accu.  |   Resolution  |  Train T. (ms/step)  |  Train FPS  |   PSNR↑    | SSIM↑  |
|:---------------|:---------------|:--------------|:-----------------------:|:----------:|:------------:|:----------------:|:----------------:|:----------------:|
| VAE-kl-f8-ema    |    D910\*x1-MS2.2.10       |      FP32   |      12x1    |    256x256  |    700      |  17.14   |   32    |  0.89    |
| VAE-kl-f8    |    G3090x1-MS2.3       |      FP32   |      4x1    |    256x256  | 800      |   5  |    32.37   |  0.90    |
> Context: {G:GPU, D:Ascend}{chip type}-{number of NPUs}-{mindspore version}.


Here are some reconstruction results (left is ground-truth, right is the reconstructed)


<p float="center">
<img src=https://github.com/SamitHuang/mindone/assets/8156835/ec7ceee8-13e0-4358-8a8a-8b5a3a3daa57 width="30%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/595eb459-96e1-442d-9152-39e0d431ff04 width="30%" />
<img src=https://github.com/SamitHuang/mindone/assets/8156835/aecc813a-71e2-4a30-971a-061f82b63e7c width="30%" />
</p>


## Causal 3D AutoEncoder

**NOTE:** To run on Ascend 910b, mindspore 2.3.0rc1+20240409 or later version is required.

### Inference

1. Download the causal vae 3d model checkpoint from HF [Open-Sora-Plan-v1.0.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/vae)


2. Convert checkpoint to mindspore format by running:

```shell
python tools/convert_vae.py --src /path/to/diffusion_pytorch_model.safetensors  --target models/causal_vae_488.ckpt
```

3. Run inference script

```shell
python infer.py --model_config configs/causal_vae_488.yaml \
    --ckpt_path models/causal_vae_488.ckpt \
    --data_path datasets/mixkit \
    --dataset_name video \
    --dtype fp16  \
    --size 512 \
    --crop_size 512 \
    --frame_stride 1 \
    --num_frames 33 \
    --batch_size 1 \
    --output_path samples/causal_vae_recons \
```

After running, it will save the reconstruction results in `samples/causal_vae_recons` and report the PSNR and SSIM evaluation metrics by default.

For detailed arguments, please run `python infer.py -h`.

NOTE: for `dtype`, only fp16 is supported on 910b+MS currently due to Conv3d. Conv3d bf16 precision will be supported later.

Here are some results.


### Training

1. Inflate the 2D vae (e.g. from stable-diffusion) for causal vae 3d

```
python tools/inflate_vae2d_to_vae3d.py --src /path/to/vae_2d.ckpt  --target models/causal_vae_488_init.ckpt
```

2. Run training by
```
python train.py --config configs/training/causal_vae_video.yaml
```

For detailed arguments, please run `python train.py -h`.

It's easy to config the model architecture in `configs/causal_vae_488.yaml` and the training strategy in `configs/training/causal_vae_video.yaml`.

### Results

The training task is under progress. The initial training performance without further optimization tuning is as follows.

| Model          |   Context   |  Precision         | Local BS x Grad. Accu.  |   Resolution  |  Train T. (ms/step)  |
|:---------------|:---------------|:--------------|:-----------------------:|:----------:|:------------:|
| causal_vae_488 |    D910\*x1-MS2.3(20240409)       |      FP16   |      1x1    |    256x256x17  |    3280
> Context: {G:GPU, D:Ascend}{chip type}-{number of NPUs}-{mindspore version}.
