# Movie Gen Video based on MindSpore


## Temporal Autoencoder (TAE) 


### Requirements

| mindspore  | ascend driver  |  firmware   |cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   2.3.1    |    24.1.RC2    | 7.3.0.1.231 |   8.0.RC2.beta1    |

### Prepare weights

We use SD3.5 VAE to initialize the spatial layers of TAE, considering they have the same number of latent channels, i.e. 16.

1. Download SD3.5 VAE from [huggingface](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main/vae)

2. Inflate VAE checkpoint for TAE initialization by

```shell
python inflate_vae_to_tae.py --src /path/to/sd3.5_vae/diffusion_pytorch_model.safetensors --target models/tae_vae2d.ckpt
```

### Prepare datasets

We need to prepare a csv annotation file listing the path to each input video related to the root folder, indicated by the `video_folder` argument. An example is
```
video
dance/vid001.mp4
dance/vid002.mp4
dance/vid003.mp4
...
```

Taking UCF-101 for example, please download the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset and extract it to `datasets/UCF-101` folder.


### Training

TAE is trained to optimize the reconstruction loss, perceptual loss, and the outlier penalty loss (OPL) proposed in the MovieGen paper.

To launch training, please run

```shell
python scripts/train_tae.py \
--config configs/tae/train/mixed_256x256x32.yaml \
--output_path /path/to/save_ckpt_and_log \
--csv_path /path/to/video_train.csv  \
--video_folder /path/to/video_root_folder  \
```

Different from the paper, we found that OPL loss doesn't benefit the training outcome in our ablation study (reducing in lower PSNR decreased). Thus we disable OPL loss by default. You may enable it by appending `--use_outlier_penalty_loss True`

For more details on the arguments, please run `python scripts/train_tae.py --help`


### Evaluation

To run video reconstruction with the trained TAE model and evaluate the PSNR and SSIM on the test set, please run

```shell
python scripts/inference_tae.py \
--ckpt_path /path/to/tae.ckpt \
--batch_size 2 \
--num_frames 32  \
--image_size 256 \
--csv_path  /path/to/video_test.csv  \
--video_folder /path/to/video_root_folder  \
```

The reconstructed videos will be saved in `samples/recons`.

#### Performance

Here, we report the training performance and evaluation results on the UCF-101 dataset.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

| model name      |  cards | batch size | resolution |  precision | jit level |   graph compile | s/step     | PSNR | SSIM | recipe |
| :--:         | :---:   | :--:       | :--:       |  :--:       | :--:       | :--:      |:--:    | :--:   |:--:   |:--:   |
| TAE  |  1     | 1      | 256x256x32   |  bf16    |   O0  | 2 mins |   2.18     | 31.35     |   0.92       |  [config](configs/tae/train/mixed_256x256x32.yaml) |


### Usages for Latent Diffusion Models

<details>
<summary>View more</summary>

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
# z = TAE_CONFIG['scaling_factor'] * (z - TAE_CONFIG['shift_factor'])

```

For detailed arguments, please refer to the docstring in [tae.py](mg/models/tae/tae.py)

### Decoding video latent

```python

# if z is scaled, you should unscale at first:
# z = z / TAE_CONFIG['scaling_factor'] + TAE_CONFIG['shift_factor']

# z - a batch of video latent, shape (b c t h w)
x = tae.decode(z)

# for image decoding, set num_target_frames to discard the spurious frames
x = tae.decode(z, num_target_frames=1)
```

</details>
