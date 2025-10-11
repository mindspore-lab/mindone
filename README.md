# MindSpore ONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation.

ONE is short for "ONE for all"

## News
- [2025.10.11] We release [v0.4.0-rc1](https://github.com/mindspore-lab/mindone/releases/tag/v0.4.0-rc1). This release candidate brings **50+ new state-of-the-art AI models** across language, vision, and audio domains, including advanced models like Reformer, XGLM, PhiMoE, VideoMAE, and comprehensive audio processing capabilities. Test it now!
- [2025.04.10] We release [v0.3.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.3.0). More than 15 SoTA generative models are added, including Flux, CogView4, OpenSora2.0, Movie Gen 30B , CogVideoX 5B~30B. Have fun!
- [2024.11.06] [v0.2.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.2.0) is released

## Quick tour

### Installation

**Previous Release :**
To install v0.3.0, please install [MindSpore 2.5.0](https://www.mindspore.cn/install) and run:
```bash
pip install mindone
```

**Release Candidate (Latest Features):**
To try v0.4.0-rc1 with 300+ additional models diffusers/transformers API , please install [MindSpore 2.6.0/2.7.0](https://www.mindspore.cn/install) and run:
```bash
pip install mindone==0.4.0rc1
```

**Development Version:**
Alternatively, to install the latest version from the `master` branch, please run:
```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
```

### What's New in v0.4.0-rc1

🚀 **300+ New AI Models**: Including Reformer, XGLM, PhiMoE, BioGPT, and many more advanced language models and specialized architectures.

🚀 **hf diffusers/transformers support** : support hf diffusers v0.35 and hf transformers v4.50

[View Full Release Notes](https://github.com/mindspore-lab/mindone/releases/tag/v0.4.0-rc1) | [Complete Changelog](https://github.com/mindspore-lab/mindone/blob/main/CHANGELOG.md)

### Getting Started

We support state-of-the-art diffusion models for generating images, audio, and video. Let's get started using [Flux Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) as an example.

**Hello MindSpore** from **Flux**!

<div>
<img src="https://github.com/user-attachments/assets/17722b48-b6c7-44a6-b736-44b4e6d7d9d4" alt="flux_kontext" width="512" height="512">
</div>


```py
import mindspore
from mindone.diffusers import FluxKontextPipeline
from mindone.diffusers.utils import load_image
import numpy as np

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", mindspore_dtype=mindspore.bfloat16
)
prompt = "A cat holding a sign that says 'Hello MindSpore'"
image = pipe(prompt)[0][0]
image.save("sd3.png")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png").convert("RGB")
prompt = "Make Pikachu hold a sign that says 'MindSpore ONE', yarn art style, detailed, vibrant colors"
image = pipe(
    image=image,
    prompt=prompt,
    guidance_scale=2.5,
    generator=np.random.default_rng(42),
)[0][0]
image.save("flux-kontext.png")
```

### supported models under mindone/examples
| model  | inference | finetune | pretrain | organization  | 
 :---    |  :---:    |  :---:   |  :---:   |  :--       |
| [sparktts](https://github.com/mindspore-lab/mindone/blob/master/examples/sparktts) | ✅ | ✖️ | ✖️ | HKUST |
| [hunyuanvideo-i2v](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo-i2v)  |  ✅  | ✖️  | ✖️  | Tencent |
| [wan2.1](https://github.com/mindspore-lab/mindone/blob/master/examples/wan2_1)  |  ✅  |  ✖️  |  ✖️   | Alibaba  |
| [wan2.2](https://github.com/mindspore-lab/mindone/blob/master/examples/wan2_2)  |  ✅  |  ✖️  |  ✖️   | Alibaba  |
| [cogview4](https://github.com/mindspore-lab/mindone/blob/master/examples/cogview)  | ✅ | ✖️  | ✖️  | Zhipuai |
| [step_video_t2v](https://github.com/mindspore-lab/mindone/blob/master/examples/step_video_t2v)  | ✅   | ✖️  | ✖️   | StepFun  |
| [flux](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_flux.md)  | ✅ | ✅ | ✖️  | Black Forest Lab |
| [omnigen](https://github.com/mindspore-lab/mindone/blob/master/examples/omnigen) | ✅ | ✅ | ✖️ | ByteDance |
| [omnigen2](https://github.com/mindspore-lab/mindone/blob/master/examples/omnigen2) | ✅ | ✅| ✖️ | ByteDance |
| [mmada](https://github.com/mindspore-lab/mindone/blob/master/examples/mmada) | ✅ | ✅ | ✅ | ByteDance |
| [janus](https://github.com/mindspore-lab/mindone/blob/master/examples/janus)   | ✅  | ✅  | ✅  | DeepSeek |
| [emu3](https://github.com/mindspore-lab/mindone/blob/master/examples/emu3)   | ✅  | ✅  | ✅  |  BAAI |
| [var](https://github.com/mindspore-lab/mindone/blob/master/examples/var) | ✅  | ✅  | ✅  | ByteDance  |
| [hpcai open sora 1.2/2.0](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_hpcai)   | ✅ | ✅ | ✅ | HPC-AI Tech  |
| [cogvideox 1.5 5B~30B ](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/cogvideox_factory)  | ✅ |  ✅  | ✅  | Zhipu |
| [open sora plan 1.3](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_pku)  | ✅ | ✅ | ✅ | PKU |
| [hunyuanvideo](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo) | ✅  | ✅  | ✅  | Tencent  |
| [movie gen 30B](https://github.com/mindspore-lab/mindone/blob/master/examples/moviegen)  | ✅ | ✅ | ✅ | Meta |
