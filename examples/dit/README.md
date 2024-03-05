# Scalable Diffusion Models with Transformers (DiT)

## Introduction
Previous common practices of diffusion models (e.g., stable diffusion models) used U-Net backbone, which lacks scalability. DiT is a new class diffusion models based on transformer architecture. The authors designed Diffusion Transformers (DiTs), which adhere to the best practices of Vision Transformers (ViTs) [<a href="#references">1</a>]. It accepts the visual inputs as a sequence of visual tokens through "patchify", and then processed the inputs  by a sequence of transformer blocks (DiT blocks). The structure of DiT model and DiT blocks is shown below:

<p align="center">
  <img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/DiT_structure.PNG" width=550 />
</p>
<p align="center">
  <em> Figure 1. The Structure of DiT and DiT blocks. [<a href="#references">2</a>] </em>
</p>

DiTs are scalable architectures for diffusion models. The authors found that there is a strong correlation between the network complexity (measured by Gflops) vs. sample quality (measured by FID). In other words, the more complex the DiT model is, the better it performs on image generation.

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

To run inference of `DiT-XL/2` model with the `256x256` image size on Ascend devices, you can use:
```bash
python sample.py -c configs/inference/image/dit-xl-2-256x256.yaml
```

To run inference of `DiT-XL/2` model with the `512x512` image size on Ascend devices, you can use:
```bash
python sample.py -c configs/inference/image/dit-xl-2-512x512.yaml
```

To run the same inference on GPU devices, simply set `--device_target GPU` for the commands above.

For diffusion sampling, the default sampler is the DDIM sampler, and the default number of sampling steps is 50. For classifier-free guidance, the default guidance scale is $4.0$. Check more details in the yaml files under `configs/inference/image/`.

Some generated example images are shown below:
<p float="center">
<img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/512x512/class-207.png" width="25%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/512x512/class-360.png" width="25%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/512x512/class-417.png" width="25%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/512x512/class-979.png" width="25%" />
</p>
<p float="center">
<img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-207.png" width="12.5%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-279.png" width="12.5%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-360.png" width="12.5%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-387.png" width="12.5%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-417.png" width="12.5%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-88.png" width="12.5%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-974.png" width="12.5%" /><img src="https://raw.githubusercontent.com/wtomin/mindone-assets/main/dit/256x256/class-979.png" width="12.5%" />
</p>

## Training

Now, we support finetuning DiT model on a toy dataset `imagenet_samples/images/`. It consists of three sample images randomly selected from ImageNet dataset and their corresponding class labels. This toy dataset is store at this [website](https://github.com/wtomin/mindone-assets/tree/main/dit/imagenet_samples). You can also download this toy dataset using:

```bash
bash scripts/download_toy_dataset.sh
```
Afterwards, the toy dataset is saved in `imagenet_samples/` folder.

To run finetuning experiments on Ascend devices, use:
```bash
python train.py --config configs/training/image/class_cond_finetune.yaml
```
You can adjust the hyper-parameters in the yaml file:
```yaml
# training hyper-params
start_learning_rate: 5e-5  # small lr for finetuing exps. Change it to 1e-4 for regular training tasks.
scheduler: "constant"
warmup_steps: 10
train_batch_size: 2
gradient_accumulation_steps: 1
weight_decay: 0.01
epochs: 3000
```

After training, the checkpoints will be saved under `output_folder/ckpt/`.

To run inference with a certain checkpoint file, please first revise `dit_checkpoint` path in the yaml files under `configs/inference/image/`, for example,
```
# dit-xl-2-256x256.yaml
dit_checkpoint: "outputs/ckpt/DiT-3000.ckpt"
```

Then run `python sample.py -c config-file-path`.


**Experimental Features**:

- We also support finetuning VideoDiT model on a toy dataset `imagenet_samples/videos/`. It consists of three videos generated using VideoLDM (`examples/svd`) based on the three images in `imagenet_samples/images/`. To run finetuning experiments on Ascend devices, use:
```bash
python train_video.py --config configs/training/video/class_cond_finetune.yaml
```


# References

[1] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020. 1, 2, 4, 5

[2] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023
