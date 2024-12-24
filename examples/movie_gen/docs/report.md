

## Temporal Autoencoder (TAE)

TAE is used to encode the RGB pixel-space videos and images into a spatio-temporally compressed latent space. In particular, the input is compressed by 8x across each spatial dimension H and W, and the temporal dimension T. We follow the framework of Meta Movie Gen [[1](#references)] as below.

<p align="center"><img width="700" alt="TAE Framework" src="https://github.com/user-attachments/assets/678c2ce6-28b8-4bda-b8a3-fac921595b8a"/>
<br><em> Videoen Eoding and Decoding using TAE </em></p>

TAE inflates an image autoencoder by adding 1-D temporal convolution in resnet blocks and attention blocks. Temporal compression is done by injecting temporal downsample and upsample layers.


### Key design & implementation

In this section, we explore the design and implementation details not illustrated in the Movie Gen paper. For example, how to perform padding and initialization for the Conv 2.5-D layers and how to configure the training frames.

#### SD3.5 VAE as the base image encoder

In TAE, the number of channels of the latent space is 16 (C=16). It can help improve both the reconstruction and the generation performance compared to C=4 used in OpenSora or  SDXL vae.

We choose to use the [VAE]() in Stable Diffusion 3.5 as the image encoder to build TAE for it has the same number of latent channels and can generalize well in image generation. 


#### Conv2.5d implementation

Firstly, we replace the Conv2d in VAE with Conv2.5d, which consists of a 2D spatial convolution followed by a 1D temporal convolution.

For 1D temporal convolution, we set kernel size 3, stride 1, symmetric replicate padding with padding size (1, 1), and input/output channels the same as spatial conv. We initialize the kernel weight so as to preserve the spatial features (i.e. preserve image encoding after temporal initialization). Therefore, we propose to use `centric` initialization as illustrated below.  

```python
w = self.conv_temp.weight
ch = int(w.shape[0])
value = np.zeros(tuple(w.shape))
for i in range(ch):
    value[i, i, 0, 1] = 1
w.set_data(ms.Tensor(value, dtype=ms.float32))
```
#### Temporal Downsampling


Paper: "Temporal downsampling is performed via strided convolution with a stride of 2". 

Our implementation: the strided convolution is computed using conv1d of kernel size 3, stride 2, and symmetric replicate padding. `centric` initialization (as mentioned in the above conv2.5 section) is used to initialize the conv kernel weight.

To achieve 8x temporal compression, we apply 3 temporal downsampling layers, each placed after the spatial downsampling layer in the first 3 levels. 

#### Temporal Upsampling
Paper: "upsampling by nearest-neighbor interpolation followed by convolution"

Our design:
1. nearest-neighbour interpolation along the temporal dimension  
2. conv1d: kernel size 3, stride 1, symmetric replicate padding, and `centric` initialization.

To achieve 8x temporal compression, we apply 3 temporal upsampling layers, each placed after the spatial upsampling layer of the last 3 levels. 



### Evaluation

We conduct experiments to verify our implementation's effectiveness on the [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) dataset containing 13,320 videos. We split the videos into training and test sets by 8:2. Here are the training performance and test accuracy. 

| model name      |  cards | batch size | resolution |  precision |   OPL Loss | s/step     | PSNR | SSIM |
| :--:         | :---:   | :--:       | :--:        | :--:       | :--:      |:--:    | :--:   |:--:   |
| TAE  |  1     | 1      | 256x256x32   |  bf16       | OFF |   2.18     | 31.35     |   0.92       | 
| TAE  |  1     | 1      | 256x256x32   |  bf16       | ON |   2.18     | 31.17     |   0.92       |


The hyper-parameters we used are as follows.

```yaml
kl loss weight:  1.0e-06
perceptual and reconstruction loss weight:  1.0
outlier penalty loss weight: 1.0
optimizer: adamw
learning rate: 1e-5
```

Here is the comparison between the origin videos (left) and the videos reconstructed with the trained TAE model (right).


<p float="center">
<img src=https://github.com/user-attachments/assets/ba3362e4-2210-4811-bedf-f19316f511d3 width="45%" />
<img src=https://github.com/user-attachments/assets/36257aef-72f0-4f4f-8bd3-dc8fb0a33fd8 width="45%" />
</p>

We further fine-tune the TAE model on the mixkit dataset, a high-quality video dataset in 1080P resolution. Here are the results.

<p float="center">
<img src=https://github.com/user-attachments/assets/7978489b-508b-4204-a4d7-d11dda3f905c width="45%" />
<img src=https://github.com/user-attachments/assets/e87105d9-1ff1-4a4c-bbfb-e07615f0fe6d width="45%" />
</p>

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] The Movie Gen team @ Meta. Movie Gen: A Cast of Media Foundation Models. 2024
