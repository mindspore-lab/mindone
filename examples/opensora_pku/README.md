# Open-Sora-Plan with MindSpore

This repository contains MindSpore implementation of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), including the autoencoders and diffusion models with their training and inference pipelines.

## Features
- [ ] Open-Sora-Plan v1.0.0
    - [x] Causal 3D Autoencoder (Video AutoEncoder)
        - [x] Inference
        - [x] Training (experimental)
    - [ ] Latte Text-to-Video 
        - [ ] Inference
        - [x] Training

## Installation

```
pip install -r requirements.txt
```

## Causal Video VAE 

**NOTE:** To run VAE 3D on Ascend 910b, mindspore 2.3.0rc1+20240409 or later version is required.

### Inference

1. Download original model checkpoint from HF [Open-Sora-Plan-v1.0.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/vae)


2. Convert the checkpoint to mindspore format:

```shell
python tools/model_conversion/convert_vae.py --src /path/to/diffusion_pytorch_model.safetensors  --target models/ae/causal_vae_488.ckpt
```

3. Video reconstruction

```shell
python infer_ae.py --model_config configs/ae/causal_vae_488.yaml \
    --ckpt_path models/ae/causal_vae_488.ckpt \
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

After running, the reconstruction results will be saved in `--output_path` and evaluated w.r.t. PSNR and SSIM metrics.

For detailed arguments, please run `python infer.py -h`.

NOTE: for `dtype`, only fp16 is supported on 910b+MS currently, due to Conv3d operator only support fp16 currently. Bettr precision will be supported soon.

Here are some visualization results.

---------------- Placeholder   --------


### Training

#### Pretraine weights

Causal video vae can be initialized from vae 2d for better convergence. This can be done by inflating the 2d vae model checkpoint as follows 

```
python tools/model_conversion/inflate_vae2d_to_vae3d.py \
    --src /path/to/vae_2d.ckpt  \
    --target models/causal_vae_488_init.ckpt
```
> In case you lack vae 2d checkpoint in mindspore format, please use `tools/model_conversion/convert_vae.py` for model conversion, e.g. after downloading the [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)) weights.

Please also download [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) and put it under `models/ae/` for training with lpips loss.


#### Launch training
Then, you can launch training by:
```
python train.py --config configs/ae/training/causal_vae_video.yaml
```

For detailed arguments, please run `python train.py -h` and check the config yaml file.

It's easy to config the model architecture in `configs/causal_vae_488.yaml` and the training strategy in `configs/training/causal_vae_video.yaml`.


**Districuted Training**: For distributed training on multiple NPUs, please refer to this [doc](../stable_diffusion_v2/README.md#distributed-training)


### Results

The training task is under progress. The initial training performance without further optimization tuning is as follows.

| Model          |   Context   |  Precision         | Local BS x Grad. Accu.  |   Resolution  |  Train T. (ms/step)  |
|:---------------|:---------------|:--------------|:-----------------------:|:----------:|:------------:|
| causal_vae_488 |    D910\*x1-MS2.3(20240409)       |      FP16   |      1x1    |    256x256x17  |    3280
> Context: {G:GPU, D:Ascend}{chip type}-{number of NPUs}-{mindspore version}.


## Video Diffusion Transformer

TBC
