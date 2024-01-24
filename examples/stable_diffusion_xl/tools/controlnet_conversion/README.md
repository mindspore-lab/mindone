# ControlNet + SDXL model weight conversion from Diffusers to MindONE

**Step1**: Convert SDXL-base-1.0 model weight from Diffusers to MindONE, refer to [here](../../GETTING_STARTED.md#convert-pretrained-checkpoint).

**Step2**: Since ControlNet acts like a plug-in to the SDXL, we convert the ControlNet weight `diffusion_pytorch_model.safetensors` from [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/tree/main)
to MindSpore version and then merge it into the SDXL-base-1.0 MindONE model weight (`sd_xl_base_1.0_ms.ckpt`, by default). Eventually, we get the ControlNet + SDXL-base-1.0 MindONE model weight (`sd_xl_base_1.0_controlnet_canny_ms.ckpt`, by default).


```shell
cd tools/controlnet_conversion

python convert_weight.py  \
    --weight_torch_controlnet /PATH TO/diffusion_pytorch_model.safetensors  \
    --weight_ms_sdxl /PATH TO/sd_xl_base_1.0_ms.ckpt  \
    --output_ms_ckpt_path /PATH TO/sd_xl_base_1.0_controlnet_canny_ms.ckpt
```

> Note: The ControlNet weight parameters name mapping between Diffusers and MindONE is prepared: `tools/controlnet_conversion/controlnet_ms2torch_mapping.yaml`.
