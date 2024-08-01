# VAE Instructions

## Prepare Pretraiend Weights

### Stage 1 training
To train VAE-3D introduced in OpenSora v1.2 from scratch (stage 1), please download the VAE-2D checkpoint for model initialization from [here](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae)

Then convert the torch checkpoint to mindspore format with the following script:

```
python tools/convert_vae1.2.py --src /path/to/pixart_sigma_sdxlvae_T5_diffusers/vae/diffusion_pytorch_model.safetensors --target models/sdxl_vae.ckpt --from_vae2d
```
