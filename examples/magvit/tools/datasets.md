# Datasets

Here we present the download and detailed preprocessing tutorial for the data we used in our training.


## Image Dataset for pretraining

Following the original paper, we use [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/index.php) to pretrain VQVAE-2D as the initialiation.

| Dataset | Train | Val |
| --- | --- | --- |
| ImageNet-1K | 1281167 | 50000 |

### Download
You can download through the link on [Datasets: ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

### Preprocessing
After downloading, please unzip the folders and rearrange them into train and validation folders:

```
├─ImageNet
│  ├─train
|  | ├─n01440764
|  | | ├─n01440764_10026.JPEG
|  | | ├─n01440764_10027.JPEG
|  │ | └─ ...
|  | ├─n01443537
|  | | ├─n01443537_10026.JPEG
|  | | ├─n01443537_10027.JPEG
|  │ | └─ ...
│  └─validation
|    ├─val_00000001.JPEG
|    ├─val_00000002.JPEG
|    └─ ...
```

## Video Dataset

In this repositry, we use [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) to train the VQVAE-3d.

We use the Train/Test Splits for Action Recognition, the statistics are:

| Dataset | Train | Test |
| --- | --- | --- |
| UCF-101| 9537 | 3783 |

### Download
You can download the dataset and *The Train/Test Splits for Action Recognition* on the page of [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php)

### Preprocessing
After downloading, please split the dataset into train and test according to the Splits text file. We also provide a script for the seperation, you can refer to [ucf101.py](./ucf101.py)
