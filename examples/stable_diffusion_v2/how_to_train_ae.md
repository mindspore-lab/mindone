# Example on training AutoEncoder with KL

### 0. Install mindone

```shell
    git clone https://github.com/mindspore-lab/mindone
    cd mindone
    pip install .
    cd mindone/examples/stable_diffusion_v2
```

### 1. download openimage data

```shell
    python download_openimage.py
```
then move downloaded csv and folders to the paths specified in `mindone/examples/stable_diffusion_v2/configs/autoencoder/autoencoder_kl_32x32x4.yaml`.

### 2. train ae

Change some training params in the yaml file. 

Note that the batch_size in the yaml file is the global batch size.

1 machine 8 cards:
```shell
    mpirun --allow-run-as-root -n 8 python train.py --is_distributed True
```
