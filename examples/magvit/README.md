# MAGVIT-v2: Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation

This folder contains the Mindspore implementation of [MAGVIT-v2](https://arxiv.org/pdf/2310.05737).

## Features

- [x] Lookup-Free-Quantization (LFQ)
- [x] VQVAE-2d Training
- [x] VQVAE-3d Training
- [ ] VQGAN Training
- [ ] MAGVIT-v2 Transformers
- [ ] MAGVIT-v2 Training

## Requirements

1. Install Mindspore >=2.3 according to the [official tutorials](https://www.mindspore.cn/install)
2. For Ascend users, please install the corresponding CANN version as stated in the official document. [CANN](https://www.mindspore.cn/install#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85)

```
pip install -r requirements.txt
```

## Datasets

Here we present an overview of the datasets we used in training. For data download and detailed preprocessing tutorial, please refer to [datasets](./tools/datasets.md)

### Image Dataset for pretraining

Following the original paper, we use [ImageNet-1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k) to pretrain VQVAE-2D as the initialiation.

| Dataset | Train | Val |
| --- | --- | --- |
| ImageNet-1K | 1281167 | 50000 |


### Video Dataset

In this repositry, we use [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) to train the VQVAE-3d.

We use the Train/Test Splits for *Action Recognition*, the statistics are:

| Dataset | Train | Test |
| --- | --- | --- |
| UCF-101| 9537 | 3783 |


## Training

### 1. VQVAE

The training of VQVAE can be divided into two stages: VQVAE-2d and VQVAE-3d, where VQVAE-2d is the initialization of VQVAE-3d.

#### 1.1 VQVAE-2d

For the pretraining of VQVAE-2d, we provide a pretrained model weights as follow:

| Model | Dataset | Image Size | Weights | PSNR | SSIM |
|-------| ------- | -----------| ------- | ------- | -------|
| VQVAE-2d | ImageNet | 128x128 | | 20.013 | 0.5734 |

If you would like you pretrain your weights, you can:

1) Prepare datasets

 We take ImageNet as an example

2) Run the training script as below:

 ```
 # standalone training
 bash scripts/run_train_vqvae_2d.sh

 # parallel training
 bash scripts/run_train_vqvae_2d_parallel.sh
 ```

3) Inflate 2d to 3d

 We provide a script for inflation, you can run the command:

 ```
 python tools/inflate_vae2d_to_3d.py --src VQVAE_2D_MODEL_PATH --target INFALTED_MODEL_PATH
 ```

#### 1.2 VQVAE-3d

Modify the path of pretrained VQVAE-2d model in [run_train_vqvae.sh](./scripts/run_train_vqvae.sh) / [run_train_vqvae_parallel.sh](./scripts/run_train_vqvae_parallel.sh)

Run the training script as below:

 ```
 # standalone training
 bash scripts/run_train_vqvae.sh

 # parallel training
  bash scripts/run_train_vqvae_parallel.sh
 ```

 The VQVAE-3d model we trained is listed below:

 | Model | Dataset | Image Size | Weights | PSNR | SSIM |
 |-------| ------- | -----------| ------- | ------- | -------|
 | VQVAE-3d | UCF-101 | 128x128 |  | 21.6529 | 0.7415 |


### 2. MAGVIT-v2

The training script of MAGVIT-v2 is still under development, so stay tuned!


## Evaluation
We provide two common evaluation metrics in our implementations: PSNR and SSIM.
To run the evaluations, you can use the following command:

```
bash scripts/run_eval_vqvae.sh
```
