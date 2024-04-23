# Flexible Vision Transformer for Diffusion Model (FiT)

*Note: This project is still work in progress.*

This folder contains [FiT](https://arxiv.org/abs/2402.12376) models implemented with [MindSpore](https://www.mindspore.cn/),

Nature is infinitely resolution-free. In the context of this reality, existing diffusion models, such as Diffusion Transformers, often face challenges when processing image resolutions outside of their trained domain. To overcome this limitation, the Flexible Vision Transformer (FiT), a transformer architecture is specifically designed for generating images with unrestricted resolutions and aspect ratios. Unlike traditional methods that perceive images as static-resolution grids, FiT conceptualizes images as sequences of dynamically-sized tokens. This perspective enables a flexible training strategy that effortlessly adapts to diverse aspect ratios during both training and inference phases, thus promoting resolution generalization and eliminating biases induced by image cropping. Enhanced by a meticulously adjusted network structure and the integration of training-free extrapolation techniques, FiT exhibits remarkable flexibility in resolution extrapolation generation. Comprehensive experiments demonstrate the exceptional performance of FiT across a broad range of resolutions, showcasing its effectiveness both within and beyond its training resolution distribution.[1]

## Get Started

### Pretrained Checkpoints

## Sampling

To run inference of `FiT-XL/2` model with the `256x256` image size on Ascend devices, you can use:
```bash
python sample.py --imagegrid True

```
You can also adjust the image size by adding the flag `--image_height` and `--image_width`. For example, you can run
```bash
python sample.py --imagegrid True --image_height 320 --image_width 160
```
to generate image with 160x320 size.

#### Intermediate Result
Some generated example images of are shown below:

<p align="center"><img width="400" src="https://github.com/zhtmike/mindone/assets/8342575/71404444-61e8-44c1-a8fb-34bed6fddb1f"/>
<br><em>Sampling Result (85/360 epochs)</em></p>


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
python preprocess.py --dataset_path PATH_TO_YOUR_DATASET --outdir latent
```

where `PATH_TO_YOUR_DATASET` is the path of your ImageNet dataset, e.g. `ImageNet2012/train`. The latent vector of each image is then stored at `latent` directory.

### Start training

You can then start the distributed training using the following command

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
mpirun -n 4 python train.py \
    -c configs/training/class_cond_train.yaml \
    --dataset_path PATH_TO_YOUR_LATENT_DATASET \
    --use_parallel True
```

where `PATH_TO_YOUR_LATENT_DATASET` is the path of directory storing the latent vectors, e.g. `./latent`.

For machine with Ascend devices, you can also start the distributed training using the rank table.
Please run

```bash
bash scripts/run_distributed.sh path_of_the_rank_table 0 4 path_to_your_latent_dataset
```

to launch a 4P training. For detail usage of the training script, please run

```bash
bash scripts/run_distributed.sh -h
```

## Benchmark

### Training

| Model    | Context       | Global Batch Size x Grad. Accu. | Max. Resolution | Acceleration | FPS (img/s) |
|----------|---------------|---------------------------------|-----------------|--------------|-------------|
| FiT-XL-2 | D910*x4-MS2.2 | 256x1                           | 256x256         | FP16         | 319.7       |

> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
> Acceleration: FP16: float16 computation. Flash attention is not used in the test currently.
> Max. Resolution: The maximum resolution of the image in training.
> FPS: images per second during training. average training time (s/step) = batch_size / FPS

### Inference

| SD Model | Context       | Scheduler | Steps | Resolution | Batch Size | Speed (step/s) |
|----------|---------------|-----------|-------|------------|------------|----------------|
| FiT-XL-2 | D910*x1-MS2.2 | DDPM      | 250   | 256x256    | 8          | 2.19           |
| FiT-XL-2 | D910*x1-MS2.2 | DDIM      | 50    | 256x256    | 8          | 1.82           |

> Context: {Ascend chip}-{number of NPUs}-{mindspore version}.
> Speed (step/s): sampling speed measured in the number of sampling steps per second.

# References

[1] Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, Lei Bai. FiT: Flexible Vision Transformer for Diffusion Model. arXiv:2402.12376, 2024.
