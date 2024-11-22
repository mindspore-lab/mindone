# Movie Gen Video


## TAE

### Requirements

ms2.3.1

### Prepare weights

We use SD3.5 VAE to initialize the spatial layers of TAE, since both have a latent channel of 16.

1. Download SD3.5 VAE from https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main/vae

2. Convert VAE checkpoint for TAE loading

```shell
python inflate_vae_to_tae.py --src /path/to/sd3.5_vae/diffusion_pytorch_model.safetensors --target models/tae_vae2d.ckpt 
```


### Training

```shell
output_dir=outputs/train_tae_256x256x16

python scripts/train_tae.py \
--config configs/tae/train/mixed_256x256x16.yaml \
--output_path=$output_dir \
--csv_path ../opensora_hpcai/datasets/mixkit-100videos/video_caption_train.csv  \
--video_folder ../opensora_hpcai/datasets/mixkit-100videos/mixkit \

```

OPL - outlier penality loss is found to be not beneficial in our experiment (PSNR decreased). Thus we set it to False by default. 

Change mixed_256x256x16.yaml to mixed_256x256x32.yaml for training on 32 frames.


#### Performance

Train on 80 samples of mixkit-100 (train set), test on the other 20 samples (test set)

256x256x16, 1p, FP32, 1.99 s/step, test set psnr 28.5

256x256x32, 1p, BF16, 2.49 s/step, test set psnr 28.3


### Inference


#### Video Reconstruction

```shell
python scripts/inference_vae.py \
--ckpt_path /path/to/tae.ckpt \
--batch_size 2 \
--num_frames=16  \
--image_size 256 \
--csv_path ../opensora_hpcai/datasets/mixkit-100videos/video_caption_test.csv  \
--video_folder ../opensora_hpcai/datasets/mixkit-100videos/mixkit \
--enable_tile=False \
```

#### Encoding video 

```python
from mg.models.tae.tae import TemporalAutoencoder, TAE_CONFIG

# may set use_tile=True to save memory
tae = TemporalAutoencoder(
    pretrained='/path/to/tae.ckpt',
    use_tile=False,
    )

# x - a batch of videos, shape (b c t h w)
z, _, _ = tae.encode(x)


# you may scale z by:
# z = TAE_CONFIG['scaling_factor'] * z + TAE_CONFIG['shift_factor'] 


```

For detailed arguments, please refer to the docstring in [tae.py](mg/models/tae/tae.py)

#### Decoding video latent

```python

# if z is scaled, you should unscale at first:
# z = (z - TAE_CONFIG['shift_factor']) / TAE_CONFIG['scaling_factor'] 

# z - a batch of video latent, shape (b c t h w)
x = tae.decode(z)

# for image decoding, set num_target_frames to discard the spurious frames 
x = tae.decode(z, num_target_frames=1)
```

