# LCM Sampler for Stable Diffusion XL (SDXL)

[Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)

Latent Consistency Models (LCM) enable quality image generation in typically 2-4 steps making it possible to use diffusion models in almost real-time settings.

## Dependency

- mindspore 2.2
- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run

```shell
pip install -r requirements.txt
```

## Preparation

### Convert Pretrained Checkpoint

We provide a script for converting pre-trained weight from `.safetensors` to `.ckpt` in `tools/model_conversion/convert_weight.py`.

step1. Download the [Official](https://github.com/Stability-AI/generative-models) pre-train weights [text_encoder](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/text_encoder), [text_encoder_2](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/text_encoder_2), [vae](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/vae) and [
lcm-sdxl](https://huggingface.co/latent-consistency/lcm-sdxl/tree/main) from huggingface.

step2. Convert a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.

```shell
cd tools/model_conversion

python convert_diffusers_to_original_sdxl.py \
  --model_path /PATH TO THE MODEL TO CONVERT \
  --checkpoint_path /PATH TO THE OUTPUT MODEL/sd_xl_base_1.0.safetensors \
  --use_safetensors \
  --unet_name "diffusion_pytorch_model.fp16.safetensors" \
  --vae_name "diffusion_pytorch_model.fp16.safetensors" \
  --text_encoder_name "model.fp16.safetensors" \
  --text_encoder_2_name "model.fp16.safetensors"
```

step3. Convert weight to MindSpore `.ckpt` format and put it to `./checkpoints/`.

```shell
cd tools/model_conversion

# convert sdxl-base-1.0 model
python convert_weight.py \
  --task pt_to_ms \
  --weight_safetensors /PATH TO/sd_xl_base_1.0.safetensors \
  --weight_ms /PATH TO/sd_xl_base_1.0_ms.ckpt \
  --key_torch torch_key_base_lcm.yaml \
  --key_ms mindspore_key_base_lcm.yaml
```

## Inference

We provide a demo for text-to-image sampling in `demo/sampling_without_streamlit.py`.

After obtaining the weights, place them into checkpoints/. We recommend setting *guidance scale* to 1.5. Next, start the demo using

```shell
# run sdxl-base txt2img without streamlit on Ascend
export MS_PYNATIVE_GE=1
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
  --device_target Ascend \
  --sampler LCMSampler \
  --sample_step 4 \
  --guidance_scale 1.5
```









