# MAGVIT-v2: Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation

This folder contains the Mindspore implementation of [MAGVIT-v2](https://arxiv.org/pdf/2310.05737). Since the official implementation is **NOT open-sourced**, we refer to the following repository implementations:
- [MAGVIT-v1](https://github.com/google-research/magvit)
- [magvit2-pytorch](https://github.com/lucidrains/magvit2-pytorch)
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)

Thanks for their great work.

## Features

- [x] Lookup-Free-Quantization (LFQ)
- [x] VQVAE-2d Training
- [x] VQVAE-3d Training
- [x] VQGAN Training


## Requirements

| mindspore | ascend driver | firmware | cann toolkit/kernel |
| --------- | ------------- | -------- | ------------------- |
| [2.3.1](https://www.mindspore.cn/)  | 24.1.RC2 |7.3.0.1.231 |	[`CANN 8.0.RC2.beta1`](https://www.hiascend.com/software/cann) |


#### Installation Tutorials:

1. Install Mindspore==2.3.1 according to the [official tutorials](https://www.mindspore.cn/install)
2. Ascend users please install the corresponding *CANN 8.0.RC2.beta1* in [community edition](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.beta1) as well as the relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers/community), as stated in the [official document](https://www.mindspore.cn/install/#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85).
3. Install the pacakges listed in requirements.txt with `pip install -r requirements.txt`


## Datasets

Here we present an overview of the datasets we used in training. For data download and detailed preprocessing tutorial, please refer to [datasets](./tools/datasets.md)

### Image Dataset for pretraining

Following the original paper, we use [ImageNet-1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k) to pretrain VQVAE-2d as the initialiation.

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

### 1. Visual Tokenizer: VQVAE

The training of VQVAE can be divided into two stages: VQVAE-2d and VQVAE-3d, where VQVAE-2d is the initialization of VQVAE-3d.

#### 1.1 2D Tokenizer

We pretrained a VQVAE-2d model using [ImageNet-1K](https://huggingface.co/datasets/ILSVRC/imagenet-1k), and the accuracy is as follows:

| Model | Token Type | #Tokens | Dataset | Image Size | Codebook Size | PSNR | SSIM |
|-------| -----------| --------| ------- | -----------| --------------| -----| -----|
| MAGVIT-v2 | 2D | 16x16 |ImageNet | 128x128 | 262144 | 20.013 | 0.5734 |

You can pretrain your model by following these steps:

1) Prepare datasets

We take ImageNet as an example. You can refer to [datasets-ImageNet](./tools/datasets.md#image-dataset-for-pretraining) to download and process the data.


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

#### 1.2 3D Tokenizer

Modify the path of `--pretrained` VQVAE-2d model in [run_train_vqvae.sh](./scripts/run_train_vqvae.sh) / [run_train_vqvae_parallel.sh](./scripts/run_train_vqvae_parallel.sh)

Run the training script as below:

 ```
 # standalone training
 bash scripts/run_train_vqvae.sh

 # parallel training
 bash scripts/run_train_vqvae_parallel.sh
 ```

 The VQVAE-3d model we trained is as follows:

| Model | Token Type | #Tokens | Dataset | Video Size | Codebook Size | PSNR | SSIM |
|-------| -----------| ------- | ------- | -----------| --------------| -----| -----|
| MAGVIT-v2 | 3D | 5x16x16 | UCF-101 | 17x128x128 | 262144 | 21.6529 | 0.7415 |


### 2. MAGVIT-v2 generation model

The training script of MAGVIT-v2 generation model is still under development, so stay tuned!


## Evaluation
We provide two common evaluation metrics in our implementations: PSNR and SSIM.
To run the evaluations, you can use the command: `bash scripts/run_eval_vqvae.sh`.

Please modify the `scripts/run_eval_vqvae.sh` accordingly as shown below:

```
# To evaluate 2D Tokenizer
--model_class vqvae-2d \
--data_path IMAGE_DATA_FOLDER \
--ckpt_path MODEL_PATH

# To evaluate 3D Tokenizer
--model_class vqvae-3d \
--data_path VIDEO_DATA_FOLDER \
--ckpt_path MODEL_PATH
```
