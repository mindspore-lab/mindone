# Training 3D-VAE

## Dataset Preparation

To train the 3D-VAE, only videos are required. Therefore, you should prepare two inputs:
1.  An input video folder that contains all the videos to be used for training. We will recurseively be search for videos.
2.  A dataset file (.csv file) that specifis the video paths (optional).

The dataset file should contain at least one column named `video`:

|video|
| ---|
|Dogs/dog1.mp4|
|Cats/cat1.mp4|

Note that the video paths in the csv file (if provided) are relative to the input video folder.


Taking UCF-101 for example, please download the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset and extract it to `datasets/UCF-101` folder. You can generate the csv annotation by running  `python tools/annotate_vae_ucf101.py`. It will result in two csv files, `datasets/ucf101_train.csv` and `datasets/ucf101_test.csv`, for training and testing respectively.


## Checkpoint Preparation

To train with LIPIPS loss, please download the [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) checkpoint and put it under `pretrained/`.

# Training Parameters

In configuration file, such as `configs/vae/train/ucf101_256x256x49.yaml`, you can set the following parameters respective to the training losses:
```yaml
  losses:
    lpips_ckpt_path: "pretrained/lpips_vgg-426bf45c.ckpt"
    disc_start: 1000
    disc_weight: 0.05
    kl_weight: 1e-6
    perceptual_weight: 0.1
    loss_type: "l1"
    print_losses: False
```

This indicates that the LPIPS loss is used and weight is 0.1. The discriminator is used after 1000 steps and the weight is 0.5. The weight of the KL loss is 1e-6. The loss type is L1, and the weight is 1. This correspondings to this training loss function:

$Loss = L_1  + 0.1 L_{lpips} + 0.05 L_{adv} + 10^{-6} L_{kl}$

# Training scripts

For standalone training, we provide a toy example: `scripts/vae/train_webvid5.sh`.

For parallel training, we provide a data parallel training script: `scripts/vae/train_ucf101.sh`. The configuration file is `configs/vae/train/ucf101_256x256x49.yaml`
