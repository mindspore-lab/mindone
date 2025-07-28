<p align="center">
  <img src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen2/refs/heads/main/assets/brand.png" width="65%">
</p>

# OmniGen2 (MindSpore)

Efficient MindSpore implementation of [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2): a unified multimodal image
generation and editing framework supporting text-to-image, instruction-guided image editing, and in-context generation.

## Overview

**OmniGen2** is a powerful and efficient generative model. Unlike OmniGen v1, OmniGen2 features two distinct decoding
pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. OmniGen2 has
competitive performance across four primary capabilities:

- **Visual Understanding**: Inherits the robust ability to interpret and analyze image content from its Qwen-VL-2.5
  foundation.
- **Text-to-Image Generation**: Creates high-fidelity and aesthetically pleasing images from textual prompts.
- **Instruction-guided Image Editing**: Executes complex, instruction-based image modifications with high precision,
  achieving state-of-the-art performance among open-source models.
- **In-context Generation**: A versatile capability to process and flexibly combine diverse inputs—including humans,
  reference objects, and scenes—to produce novel and coherent visual outputs.

## News

- MindSpore inference pipeline and Gradio demo are available under `examples/omnigen2/`.
- Example presets are provided via `configs/app.yaml` and support URL-based images.

## Requirements

| MindSpore | Ascend Driver |  Firmware   | CANN toolkit/kernel |
|:---------:|:-------------:|:-----------:|:-------------------:|
|   2.6.0   |   24.1.rc3    | 7.7.0.1.238 |       8.1.RC1       |

1) Install MindSpore and Ascend software per the official docs:

    - CANN 8.1.RC1: https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1
    - MindSpore: https://www.mindspore.cn/install/

2) Install Python dependencies:

    ```shell
    pip install -r requirements.txt
    ```

## Demo

[//]: # (TODO)

## Model Weights

OmniGen2 weights and assets are hosted on Hugging Face.

```shell
hf download OmniGen2/OmniGen2 --exclude "assets/*"
```

> [!TIP]
> For users in Mainland China set the `HF_ENDPOINT=https://hf-mirror.com` environment variable.

## Inference

### Command Line

Usage examples are available under `scripts/run`. These scripts provide ready-to-use commands for common inference
scenarios and demonstrate various OmniGen2 capabilities.

For full list of flags, see `python scripts/inference.py --help`.

#### Speedup Inference with Caching

- For TeaCache (~30% speedup at default threshold), add the following flags:

```shell
--enable_teacache --teacache_rel_l1_thresh 0.05
```

- For TaylorSeer (up to ~2× speedup, mutually exclusive with TeaCache):

```shell
--enable_taylorseer
```

### Gradio App

A local demo UI is available at `app.py`.

```shell
pip install gradio
python app.py
```

## Usage Tips

To achieve optimal results with OmniGen2, you can adjust the following key hyperparameters based on your specific use
case.

- **Guidance scales**
    - `text_guidance_scale`: stronger adherence to text (default 5.0)
    - `image_guidance_scale`: stronger adherence to input images (edit/in-context). Try 1.2–2.0 for editing; 2.5–3.0 for
      in-context.
- **Scheduler**: `euler` (default) or `dpmsolver++` for potentially fewer steps at similar quality.
- **CFG range**: Lower `--cfg_range_end` can reduce latency with minor quality impact.
- **Prompts**: Be specific. English prompts work best currently. Longer, descriptive prompts often help.
- **Inputs**: Prefer clear images ≥ 512×512.

## Training

Coming soon

## Acknowledgement

If you find OmniGen2 useful, please cite the original work:

```bibtex
@article{wu2025omnigen2,
  title={OmniGen2: Exploration to Advanced Multimodal Generation},
  author={Chenyuan Wu and Pengfei Zheng and Ruiran Yan and Shitao Xiao and Xin Luo and Yueze Wang and Wanli Li and Xiyan Jiang and Yexin Liu and Junjie Zhou and Ze Liu and Ziyi Xia and Chaofan Li and Haoge Deng and Jiahao Wang and Kun Luo and Bo Zhang and Defu Lian and Xinlong Wang and Zhongyuan Wang and Tiejun Huang and Zheng Liu},
  journal={arXiv preprint arXiv:2506.18871},
  year={2025}
}
```
