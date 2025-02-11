# Inference

<img src="https://github.com/mindspore-lab/mindone/assets/20476835/68d132e1-a954-418d-8cb8-5be4d8162342" width="200" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/9f0d0d2a-2ff5-4c9b-a0d0-1c744762ee92" width="200" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/dbaf0c77-d8d3-4457-b03c-82c3e4c1ba1d" width="200" />
<img src="https://github.com/mindspore-lab/mindone/assets/20476835/f52168ef-53aa-4ee9-9f17-6889f10e0afb" width="200" />

| [Model] SDXL-Base | [Model] SDXL-Pipeline | [Model] SDXL-Refiner | [Func] Samplers | [Func] Flash Attn |
|:-----------------:|:---------------------:|:--------------------:|:---------------:|:-----------------:|
| ✅                 | ✅                   | ✅                  | 7 samplers       | ✅                |

## Requirements

| mindspore      | ascend driver | firmware    | cann toolkit/kernel |
|:-------------:|:-------------:|:-----------:|:-------------------:|
| 2.3.1    | 24.1.RC2      | 7.3.0.1.231 | 8.0.RC2.beta1        |

## Pretrained models

Please follow SDXL [weight convertion](./preparation.md#convert-pretrained-checkpoint) for detailed steps and put the pretrained weight to `./checkpoints/`.

The scripts automatically download the clip tokenizer. If you have network issues with it, [FAQ Qestion 5](./faq_cn.md#5-连接不上huggingface-报错-cant-load-tokenizer-for-openaiclip-vit-large-patch14) helps.

## Inference

### 1. Inference with SDXL-Base

- Run with interactive visualization:

```shell
# run with streamlit
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run demo/sampling.py --server.port <your_port>
```

- Run with `sampling_without_streamlit.py` script:

```shell
# run sdxl-base txt2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
```

### 2. Inference with SDXL-Refiner

```shell
# run sdxl-refiner img2img without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task img2img \
  --config configs/inference/sd_xl_refiner.yaml \
  --weight checkpoints/sd_xl_refiner_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --img /PATH TO/img.jpg

# run pipeline without streamlit on Ascend
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --add_pipeline True \
  --pipeline_config configs/inference/sd_xl_refiner.yaml \
  --pipeline_weight checkpoints/sd_xl_refiner_1.0_ms.ckpt
```


### 4. Inference with different schedulers

A scheduler defines how to iteratively add noise to an image in training and how to update a sample based on a model’s output in inference.

SDXL uses the DDPM formulation by default, which is set in `denoiser_config`  in yaml file. See `configs/inference/sd_xl_base.yaml`. The `denoiser_config` of the model in yaml config file together with the args of samplers such as `sampler`, `guider` and `discretization` in sampling script define a scheduler in inference.

Examples of [EDM-style](https://arxiv.org/abs/2006.11239) inference are as below.

* EDM formulation of Euler sampler (EDMEulerScheduler)

  ```shell
  python demo/sampling_without_streamlit.py \
    --config configs/inference/sd_xl_base.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
    --sampler EulerEDMSampler \
    --sample_step 20 \
    --guider VanillaCFG  \
    --guidance_scale 3.0 \
    --discretization EDMDiscretization \
    --sigma_min 0.002 \
    --sigma_max 80.0 \
    --rho 7.0
  ```

* EDM formulation of DPM++ 2M sampler (EDMDPMsolverMultistepScheduler)

  ```shell
  python demo/sampling_without_streamlit.py \
    --config configs/inference/sd_xl_base.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
    --sampler DPMPP2MSampler \
    --sample_step 20 \
    --guider VanillaCFG \
    --guidance_scale 5.0 \
    --discretization EDMDiscretization \
    --sigma_min 0.002 \
    --sigma_max 80.0 \
    --rho 7.0
  ```

### 5. Inference with LCM sampler

[Latent Consistency Models (LCM)](https://arxiv.org/abs/2310.04378) enable quality image generation in typically 2-4 steps making it possible to use diffusion models in almost real-time settings.

LCM uses a different UNet weight from the offical SDXL weights of stabilityai, so we need another checkpoint conversion.

1. Download the [vae](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/vae), [text_encoder](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/text_encoder), [text_encoder_2](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/text_encoder_2) folders from the official sdxl-base, and the [unet](https://huggingface.co/latent-consistency/lcm-sdxl/tree/main) from lcm-sdxl on hugging face. Put the 4 folders in a local path as HF Diffusers format weights.

2. Then convert them into one Stable Diffusion checkpoint (safetensor format),

```shell
cd tools/model_conversion

python convert_diffusers_to_original_sdxl.py \
  --model_path /PATH_TO_THE_MODEL_TO_CONVERT \
  --checkpoint_path /PATH_TO_THE_OUTPUT_MODEL/sd_xl_base_1.0.safetensors \
  --use_safetensors \
  --unet_name "diffusion_pytorch_model.fp16.safetensors" \
  --vae_name "diffusion_pytorch_model.fp16.safetensors" \
  --text_encoder_name "model.fp16.safetensors" \
  --text_encoder_2_name "model.fp16.safetensors"
```
3. Finally, convert `safetensor` to MindSpore `.ckpt` format and put it to `./checkpoints/`.

```shell
cd tools/model_conversion

# convert sdxl-base-1.0 model
python convert_weight.py \
  --task st_to_ms \
  --weight_safetensors /PATH_TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH_TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base_lcm.yaml \
  --key_ms mindspore_key_base_lcm.yaml
```

Now we can run inference with LCM sampler,

```shell
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k" \
  --device_target Ascend \
  --sampler LCMSampler \
  --sample_step 4 \
  --guidance_scale 1.5
```

<div align="center">
<img src="https://github.com/mindspore-lab/mindone/assets/73014084/32a7fc54-c6a9-48d8-b94f-7e16b0a56cce" width="20%" />
<img src="https://github.com/mindspore-lab/mindone/assets/73014084/bf15ba17-27db-46f0-ace4-f2704bc22662" width="20%" />
<img src="https://github.com/mindspore-lab/mindone/assets/73014084/f574f17b-6ef8-4d88-8512-b0f047ad2393" width="20%" />
<img src="https://github.com/mindspore-lab/mindone/assets/73014084/ae30731a-f336-4fad-8e1b-c7f679fd3277" width="20%" />

<p align="center">
  <em> LCM Sampler 4 steps(pic 1) and EulerEDM Sampler 40 steps(pic 2) Prompt: "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k";
  LCM Sampler 4 steps(pic 3) and EulerEDM Sampler 40 steps(pic 4) Prompt: "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux" </em>
</p>
</div>

## Performance

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.
| model name    | resolution |flash attn     | sampler        | steps        | jit level  | graph compile | s/img |
|:-------------:|:---------:  |:-------------:|:--------:      |:-----------:| :--:       |:------------: |:-----------:|
| SDXL-Base     | 1024x1024   | ON            | EulerEDM       | 40          | O2         | 533.59s       | 6.78       |
| SDXL-Base     | 1024x1024   | ON            | DPM++2M Karras | 20          | O2         | 631.39s       | 3.62       |
| SDXL-Refiner  | 1024x1024   | ON            | EulerEDM       | 40          | O2         | 395.14s       | 10.18      |
| SDXL-Pipeline | 1024x1024   | ON            | EulerEDM       | 20          | O2         | 324.83s/236.9s| 5.78/2.15 |
