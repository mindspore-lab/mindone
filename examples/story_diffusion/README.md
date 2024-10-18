# StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation
This is an efficient MindSpore implementation of [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion/tree/main)

## Introduction

StoryDiffusion can generate consistent images and videos given a set of text prompts. The major contributions are two-fold:

1. Consistent self-attention, a training-free and pluggable module with all SD1.5 and SDXL-based image diffusion models. Given at least 3 text prompts, it can generate images with consistent subjects and good text-vision alignment.
2. Motion predictor for long-range video generation, which can predict motion between condition images.

The current implementation only supports text-to-image generation with consistent self-attention.

## TODOs
- [x] Comic Generation Inference script
- [x] Gradio demo of comic generation
- [ ] Motion predictor with condition images

## Installation
1. Use python>=3.8 [[install]](https://www.python.org/downloads/)

2. Please install MindSpore 2.3.1 according to the [MindSpore official website](https://www.mindspore.cn/install/) and install [CANN 8.0.RC2.beta1](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC2.2.beta1) as recommended by the official installation website.


3. Install requirements
```bash
pip install -r requirements.txt
```

## Usage

### Comic Generation

Please run the following inference script to generate consistent images:
```shell
python run.py --general_prompt "a man with a black suit" --sd_model_name "RealVision" --style_name "Comic book" --output_dir "outputs"
```
This will generate some images under the folder named "outputs". These images are conditioned on a set of text prompts, like:
```text
a man with a black suit wake up in the bed,
a man with a black suit have breakfast,
...
```

Example images are shown below:

<p align="center">
  <img src="https://github.com/wtomin/mindone-assets/blob/main/story_diffusion/0-Comic%20book-RealVision_update.png?raw=true" width=550 />
</p>

<p align="center">
  <img src="https://github.com/wtomin/mindone-assets/blob/main/story_diffusion/1-Comic%20book-RealVision_update.png?raw=true" width=550 />
</p>

Now this inference script support the following SD-XL models:
```python
models_dict = {
    "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
    "RealVision": "SG161222/RealVisXL_V4.0",
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y",
}
```
The default cache directory is `./cache_dir`. If you have the model files downloaded and stored under `./cache_dir`, you can run the inference script appened with `--local_files_only` to skip the downloading process.

### Local Gradio Demo

You can also start a local gradio demo by running:
```bash
python gradio_app_sdxl_specific_id_low_vram.py
```
