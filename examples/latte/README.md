# Latte: Latent Diffusion Transformer for Video Generation

## Introduction of Latte

Latte is a novel Latent Diffusion Transformer designed for video generation. It is built based on DiT (a diffusion transformer model for image generation). For introduction of DiT, please refer to [README of DiT](../dit/README.md).

Latte first uses a VAE (Variational AutoEncoder) to compress the video data into a latent space, and then extracts spatial-temporal tokens given the latent codes. Similar to DiT, it stacks multiple transformer blocks to model the video diffusion in the latent space. How to design the spatial and temporal blocks becomes a major question.

Through experiments and analysis, they found the best practice is (a) structure in the image below. It stacks spatial blocks and temporal blocks alternately to model spatial attentions and temporal attentions in turns.


<p align="center">
  <img src="https://raw.githubusercontent.com/Vchitect/Latte/9ededbe590a5439b6e7013d00fbe30e6c9b674b8/visuals/architecture.svg" width=550 />
</p>
<p align="center">
  <em> Figure 1. The Structure of Latte and Latte transformer blocks. [<a href="#references">2</a>] </em>
</p>

Similar to DiT, Latte supports un-conditional video generation and class-labels-conditioned video generation. In addition, it supports to generate videos given text captions.


## Get Started
In this tutorial, we will introduce how to run inference and finetuning experiments using MindONE.

### Environment Setup

```
pip install -r requirements.txt
```

`decord` is required for video generation. In case `decord` package is not available in your environment, try `pip install eva-decord`.
Instruction on ffmpeg and decord install on EulerOS:
```
1. install ffmpeg 4, referring to https://ffmpeg.org/releases
    wget wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2 --no-check-certificate
    tar -xvf ffmpeg-4.0.1.tar.bz2
    mv ffmpeg-4.0.1 ffmpeg
    cd ffmpeg
    ./configure --enable-shared         # --enable-shared is needed for sharing libavcodec with decord
    make -j 64
    make install
2. install decord, referring to https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source
    git clone --recursive https://github.com/dmlc/decord
    cd decord
    rm build && mkdir build && cd build
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
    make -j 64
    make install
    cd ../python
    python3 setup.py install --user
```

### Pretrained Checkpoints

We refer to the [official repository of Latte](https://github.com/Vchitect/Latte/tree/main) for pretrained checkpoints downloading. The pretrained checkpoint files trained on FaceForensics, SkyTimelapse, Taichi-HD and UCF101 (256x256) can be downloaded from [huggingface](https://huggingface.co/maxin-cn/Latte/tree/main).

After downloading the `{}.pt` file, please place it under the `models/` folder, and then run `tools/latte_converter.py`. For example, to convert `models/skytimelapse.pt`, you can run:
```bash
python tools/latte_converter.py --source models/skytimelapse.pt --target models/skytimelapse.ckpt
```

Please also download the VAE checkpoint from [huggingface/stabilityai.co](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main), and convert this VAE checkpoint by running:
```bash
python tools/vae_converter.py --source path/to/vae/ckpt --target models/sd-vae-ft-mse.ckpt
```

## Sampling

For example, to run inference of `skytimelapse` model with the `256x256` image size on Ascend devices, you can use:
```bash
python sample.py -c configs/inference/sky.yaml
```

## Training


# References
