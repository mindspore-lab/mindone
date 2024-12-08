# Flexible Vision Transformer for Diffusion Model (FiT)

*Note: This project is still work in progress.*

This folder contains [FiT](https://arxiv.org/abs/2402.12376) models implemented with [MindSpore](https://www.mindspore.cn/),

Nature is infinitely resolution-free. In the context of this reality, existing diffusion models, such as Diffusion Transformers, often face challenges when processing image resolutions outside of their trained domain. To overcome this limitation, the Flexible Vision Transformer (FiT), a transformer architecture is specifically designed for generating images with unrestricted resolutions and aspect ratios. Unlike traditional methods that perceive images as static-resolution grids, FiT conceptualizes images as sequences of dynamically-sized tokens. This perspective enables a flexible training strategy that effortlessly adapts to diverse aspect ratios during both training and inference phases, thus promoting resolution generalization and eliminating biases induced by image cropping. Enhanced by a meticulously adjusted network structure and the integration of training-free extrapolation techniques, FiT exhibits remarkable flexibility in resolution extrapolation generation. Comprehensive experiments demonstrate the exceptional performance of FiT across a broad range of resolutions, showcasing its effectiveness both within and beyond its training resolution distribution.[1]

## Get Started

### Requirements

| mindspore | ascend driver | firmware    | cann toolkit/kernel         |
|:---------:|:-------------:|:-----------:|:---------------------------:|
| 2.3.1     | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1               |

Python: 3.9 or later.

Then run `pip install -r requirements.txt` to install the necessary packages.

## Training with ImageNet format

We provide the training script for dataset in the ImageNet format.

### Data Preperation

You may download the ImageNet-1K data from (https://www.image-net.org/challenges/LSVRC/2012/index.php). After decompressing, the folder structure will look like this

```text
ImageNet2012/
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   └── ...
│   └── ...
└── val
    └── ...
```

We use the `train` folder for training the FiT model.

### Latent Extraction

You need first extract the latent vectors of ImageNet Dataset using the following command:

```bash
python preprocess.py --data_path PATH_TO_YOUR_DATASET --outdir latent
```

where `PATH_TO_YOUR_DATASET` is the path of your ImageNet dataset, e.g. `ImageNet2012/train`. The latent vector of each image is then stored at `latent` directory.

### Start training

You can then start the distributed training using the following command

```bash
msrun --worker_num=4 --local_worker_num=4 train.py \
    -c configs/training/class_cond_train.yaml \
    --data_path PATH_TO_YOUR_LATENT_DATASET \
    --use_parallel True
```

where `PATH_TO_YOUR_LATENT_DATASET` is the path of directory storing the latent vectors, e.g. `./latent`.

## Sampling

To run inference of `FiT-XL/2` model with the `256x256` image size on Ascend devices, you can use:
```bash
python sample.py --imagegrid True --fit_checkpoint PATH_TO_YOUR_CKPT
```
where `PATH_TO_YOUR_CKPT` is the path of your trained checkpoint.

You can also adjust the image size by adding the flag `--image_height` and `--image_width`. For example, you can run
```bash
python sample.py --imagegrid True --image_height 320 --image_width 160 --fit_checkpoint PATH_TO_YOUR_CKPT
```
to generate image with 160x320 size.

#### Intermediate Result
Some generated example images of are shown below:

<p align="center"><img width="400" src="https://github.com/zhtmike/mindone/assets/8342575/71404444-61e8-44c1-a8fb-34bed6fddb1f"/>
<br><em>Sampling Result (85/360 epochs)</em></p>

## Performance

### Training Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode

| model name | cards | batch size   | resolution  | recompute  | sink      | jit level | graph compile | s/step |         img/s |
| :--------: | :---: | :-----------:| :--------:  | :--------: | :-------: | :-------: | :-----------: | :----: | :-----------: |
| FiT-XL-2   |   4   |     64       |  256x256    |    ON      |    OFF    |    O0     |   3~5 mins    | 0.73   |    315        |

> s/step: training time measured in the number of seconds for each training step.\
> imgs/s: images per second during training. imgs/s = cards * batch_size / step time

### Inference Performance

| model name   | cards | scheduler | batch size | resolution  | jit level  |  graph compile | s/step    |
|:------------:|:-----:|:---------:|:----------:|:-----------:|:----------:|:--------------:|:---------:|
| FiT-XL-2     | 1     | IDDPM     | 8          |256 x 256    | O0         | < 3 mins       | 0.09      |
| FiT-XL-2     | 1     | DDIM      | 8          |256 x 256    | O0         | < 3 mins       | 0.09      |

# References

[1] Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, Lei Bai. FiT: Flexible Vision Transformer for Diffusion Model. arXiv:2402.12376, 2024.
