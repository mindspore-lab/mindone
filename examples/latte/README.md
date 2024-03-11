# Latte: Latent Diffusion Transformer for Video Generation

## Introduction of Latte


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
