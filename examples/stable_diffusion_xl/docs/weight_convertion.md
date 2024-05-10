## Convert Pretrained Checkpoint

We provide a script for converting pre-trained weight from `.safetensors` to `.ckpt` in `tools/model_conversion/convert_weight.py`.

step1. Download the [Official](https://github.com/Stability-AI/generative-models) pre-train weights [SDXL-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [SDXL-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) from huggingface.

step2. Convert weight to MindSpore `.ckpt` format and put it to `./checkpoints/`.

```shell
cd tools/model_conversion

# convert sdxl-base-1.0 model
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base.yaml \
  --key_ms mindspore_key_base.yaml

# convert sdxl-refiner-1.0 model
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_refiner_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_refiner_1.0_ms.ckpt \
  --key_torch torch_key_refiner.yaml \
  --key_ms mindspore_key_refiner.yaml
```

(Option) Step3. Replace and convert VAE, Download [vae-fp16-fix weights](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) from huggingface.

```shell
python convert_diffusers_to_mindone_sdxl.py \
  --model_path /PATH TO/sdxl-vae-fp16-fix \                 # dir of vae weight
  --vae_name diffusion_pytorch_model.safetensors \          # source vae weight, from https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
  --sdxl_base_ckpt /PATH TO/sd_xl_base_1.0_ms.ckpt          # base checkpoint, from Step2
  --checkpoint_path /PATH TO/sd_xl_base_1.0_vaefix_ms.ckpt  # output path
```
