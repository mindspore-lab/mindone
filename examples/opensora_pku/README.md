# Open-Sora-Plan with MindSpore

This repository contains MindSpore implementation of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), including the autoencoders and diffusion models with their training and inference pipelines.

We aim to achieve efficient training and inference on Ascend NPU devices based on MindSpore framework.

## Features
- [ ] Open-Sora-Plan v1.0.0
    - [x] Causal Video Autoencoder
        - [x] Inference
        - [x] Training (precision to be improved)
    - [ ] Latte Text-to-Video (Coming soon)
        - [ ] Inference
        - [ ] Training

## Installation

Please make sure the following frameworks are installed.

- python >= 3.8
- mindspore >= 2.3.0rc1+20240409  [[install](https://www.mindspore.cn/install)]

```
pip install -r requirements.txt
```

## Causal Video VAE

**NOTE:** To run VAE 3D on Ascend 910b, mindspore 2.3.0rc1+20240409 or later version is required.

### Inference

1. Download the original model checkpoint from HF [Open-Sora-Plan-v1.0.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/vae)


2. Convert the checkpoint to mindspore format:

```shell
python tools/model_conversion/convert_vae.py --src /path/to/diffusion_pytorch_model.safetensors  --target models/ae/causal_vae_488.ckpt
```

3. Run video reconstruction

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

NOTE: for `dtype`, only fp16 is supported on 910b+MS currently, due to Conv3d operator precision limitation. Better precision will be supported soon.

Here are some reconstruction results (left: source video clip, right: reconstructed).

![mixkit-step003-00](https://github.com/SamitHuang/mindone/assets/8156835/bb04783f-4cc1-4179-8882-940898803a6e)

![mixkit-step000-00](https://github.com/SamitHuang/mindone/assets/8156835/1582f678-55dd-4ba1-9692-4d8961a37658)

![mixkit-step002-00](https://github.com/SamitHuang/mindone/assets/8156835/f1a5e323-f3d9-4bc7-a5d2-7c6044ed52f7)



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

It's easy to config the model architecture in `configs/ae/causal_vae_488.yaml` and the training strategy in `configs/ae/training/causal_vae_video.yaml`.


**Districuted Training**: For distributed training on multiple NPUs, please refer to this [doc](../stable_diffusion_v2/README.md#distributed-training)


### Performance

The training task is under progress. The initial training performance without further optimization tuning is as follows.

| Model          |   Context   |  Precision         | Local BS x Grad. Accu.  |   Resolution  |  Train T. (ms/step)  |
|:---------------|:---------------|:--------------|:-----------------------:|:----------:|:------------:|
| causal_vae_488 |    D910\*x1-MS2.3(20240409)       |      FP16   |      1x1    |    256x256x17  |    3280
> Context: {G:GPU, D:Ascend}{chip type}-{number of NPUs}-{mindspore version}.


## Video Diffusion Transformer

### Inference

After the Causal Video VAE is prepared, you can take the following steps to run text-to-video inference.

1. Please download the model checkpoints from HF [Open-Sora-Plan-v1.0.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main). There are three stages of model checkpoints, saved in `17x256x256`, `65x256x256`, and `65x512x512` sub-folders.

Taking `17x256x256` as an example, please download the torch checkpoint from the given [URL](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.0.0/tree/main/17x256x256), and place it under `models/17x256x256/`.

2. Convert the checkpoint to mindspore format:

```shell
tools/model_conversion/convert_latte.py --src models/17x256x256/diffusion_pytorch_model.safetensors  --target models/17x256x256/model.ckpt
```

3. Run text-to-video inference.

```shell
python sample_t2v.py --config configs/diffusion/latte_17x256x256_122.yaml
```
