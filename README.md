# MindSpore ONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation.

ONE is short for "ONE for all"

## News
- [2025.09.15] We upgrade diffusers to v0.33.1 and transformers to v4.50.1 based on MindSpore. QwenImage, FluxKontext, Wan2.2, OmniGen2 and more than 20 generative models are now supported.
- [2025.04.10] We release [v0.3.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.3.0). More than 15 SoTA generative models are added, including Flux, CogView4, OpenSora2.0, Movie Gen 30B , CogVideoX 5B~30B. Have fun!
- [2025.02.21] We support DeepSeek [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-7B), a SoTA multimodal understanding and generation model. See [here](examples/janus)
- [2024.11.06] [v0.2.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.2.0) is released

## Quick tour

We recommend to install the latest version from the `master` branch based on MindSpore 2.6.0:

```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
```

We support state-of-the-art diffusion models for generating images, audio, and video. Let's get started using [Flux Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) as an example.

**Hello MindSpore** from **Flux**!
<!-- TODO: add Flux Kontext or QwenImage running result -->

<div>
<img src="https://github.com/townwish4git/mindone/assets/143256262/8c25ae9a-67b1-436f-abf6-eca36738cd17" alt="flux_kontext" width="512" height="512">
</div>

```py
import mindspore as ms
from mindone.diffusers import FluxKontextPipeline
from mindone.diffusers.utils import load_image
import numpy as np

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", mindspore_dtype=ms.bfloat16
)

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

###  run hf diffusers on mindspore
 - mindone diffusers is under active development, most tasks were tested with mindspore 2.6.0 on Ascend Atlas 800T A2 machines.
 - compatibale with hf diffusers 0.33.1. diffusers 0.35 is under development.

| component  |  features  
| :---   |  :--  
| [pipeline](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines) | support text-to-image,text-to-video,text-to-audio tasks 240+
| [models](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/models) | support audoencoder & transformers base models same as hf diffusers 50+
| [schedulers](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/schedulers) | support diffusion schedulers (e.g., ddpm and dpm solver) same as hf diffusers 35+

### supported models under mindone/examples

<!-- TODO: update the links after PR merged-->

| task | model  | inference | finetune | pretrain | institute  |
| :---   |  :---   |  :---:    |  :---:  |  :---:     |  :--  |
| Text/Image-to-Image | [qwen_image](https://github.com/mindspore-lab/mindone/pull/1288) 🔥🔥🔥 | ✅ | ✖️  | ✖️  | Alibaba |
| Text/Image-to-Image | [flux_kontext](https://github.com/mindspore-lab/mindone/blob/master/docs/diffusers/api/pipelines/flux.md) 🔥🔥🔥 | ✅ | ✖️  | ✖️  | Black Forest Labs |
| Text/Image/Speech-to-Video | [wan2_2](https://github.com/mindspore-lab/mindone/pull/1243) 🔥🔥🔥 | ✅ | ✖️  | ✖️  | Alibaba |
| Text/Image-to-Image | [omni_gen](https://github.com/mindspore-lab/mindone/blob/master/examples/omnigen) 🔥🔥 |  ✅  |  ✅  | ✖️  | Vector Space Lab|
| Text/Image-to-Image | [omni_gen2](https://github.com/mindspore-lab/mindone/blob/master/examples/omnigen2) 🔥🔥 |  ✅  |  ✖️  | ✖️  | Vector Space Lab |
| Image-to-Video | [hunyuanvideo-i2v](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo-i2v) 🔥🔥 |  ✅  | ✖️  | ✖️  | Tencent |
| Text/Image-to-Video | [wan2.1](https://github.com/mindspore-lab/mindone/blob/master/examples/wan2_1) 🔥🔥🔥 |  ✅  |  ✖️  |  ✖️   | Alibaba  |
| Text-to-Image | [cogview4](https://github.com/mindspore-lab/mindone/blob/master/examples/cogview) 🔥🔥🔥 | ✅ | ✖️  | ✖️  | Zhipuai |
| Text-to-Video | [step_video_t2v](https://github.com/mindspore-lab/mindone/blob/master/examples/step_video_t2v) 🔥🔥 | ✅   | ✖️  | ✖️   | StepFun  |
| Image-Text-to-Text | [qwen2_vl](https://github.com/mindspore-lab/mindone/blob/master/examples/qwen2_vl) 🔥🔥🔥|  ✅ |  ✖️ |  ✖️   | Alibaba |
| Any-to-Any | [janus](https://github.com/mindspore-lab/mindone/blob/master/examples/janus)  🔥🔥🔥 | ✅  | ✅  | ✅  | DeepSeek |
| Any-to-Any | [emu3](https://github.com/mindspore-lab/mindone/blob/master/examples/emu3)  🔥🔥 | ✅  | ✅  | ✅  |  BAAI |
| Class-to-Image | [var](https://github.com/mindspore-lab/mindone/blob/master/examples/var)🔥🔥 | ✅  | ✅  | ✅  | ByteDance  |
| Text/Image-to-Video | [hpcai open sora 1.2/2.0](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_hpcai)   🔥🔥   | ✅ | ✅ | ✅ | HPC-AI Tech  |
| Text/Image-to-Video | [cogvideox 1.5 5B~30B ](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/cogvideox_factory) 🔥🔥 | ✅ |  ✅  | ✅  | Zhipu  |
| Text-to-Video | [open sora plan 1.3](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_pku) 🔥🔥 | ✅ | ✅ | ✅ | PKU |
| Text-to-Video | [hunyuanvideo](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo) 🔥🔥| ✅  | ✅  | ✅  | Tencent  |
| Text-to-Video | [movie gen 30B](https://github.com/mindspore-lab/mindone/blob/master/examples/moviegen) 🔥🔥  | ✅ | ✅ | ✅ | Meta |
| Video-Encode-Decode | [magvit](https://github.com/mindspore-lab/mindone/blob/master/examples/magvit) |  ✅  |  ✅  |  ✅  | Google  |
| Text-to-Image | [story_diffusion](https://github.com/mindspore-lab/mindone/blob/master/examples/story_diffusion) | ✅  | ✖️  | ✖️  | ByteDance |
| Image-to-Video | [dynamicrafter](https://github.com/mindspore-lab/mindone/blob/master/examples/dynamicrafter)     | ✅  | ✖️  | ✖️  | Tencent  |
| Video-to-Video | [venhancer](https://github.com/mindspore-lab/mindone/blob/master/examples/venhancer) |  ✅  | ✖️  | ✖️  | Shanghai AI Lab |
| Text-to-Video | [t2v_turbo](https://github.com/mindspore-lab/mindone/blob/master/examples/t2v_turbo) |   ✅ |   ✅ |   ✅ | Google |
| Image-to-Video | [svd](https://github.com/mindspore-lab/mindone/blob/master/examples/svd) | ✅  |  ✅ | ✅  | Stability AI |
| Text-to-Video | [animate diff](https://github.com/mindspore-lab/mindone/blob/master/examples/animatediff) | ✅  | ✅  | ✅  | CUHK |
| Text/Image-to-Video | [video composer](https://github.com/mindspore-lab/mindone/tree/master/examples/videocomposer)     | ✅  | ✅  | ✅  | Alibaba |
| Text-to-Image | [flux](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_flux.md)  🔥 | ✅ | ✅ | ✖️  | Black Forest Lab |
| Text-to-Image | [stable diffusion 3](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_sd3.md) 🔥| ✅ | ✅ | ✖️ | Stability AI |
| Text-to-Image | [kohya_sd_scripts](https://github.com/mindspore-lab/mindone/blob/master/examples/kohya_sd_scripts) | ✅ | ✅ | ✖️  | kohya |
| Text-to-Image | [stable diffusion xl](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/README_sdxl.md)  | ✅ | ✅ | ✅ | Stability AI|
| Text-to-Image | [stable diffusion](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_v2) | ✅ | ✅ | ✅ | Stability AI |
| Text-to-Image | [hunyuan_dit](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuan_dit)     | ✅ | ✅ | ✅ | Tencent |
| Text-to-Image | [pixart_sigma](https://github.com/mindspore-lab/mindone/blob/master/examples/pixart_sigma)     | ✅ | ✅ | ✅ | Huawei |
| Text-to-Image | [fit](https://github.com/mindspore-lab/mindone/blob/master/examples/fit) | ✅ | ✅ | ✅ | Shanghai AI Lab  |
| Class-to-Video | [latte](https://github.com/mindspore-lab/mindone/blob/master/examples/latte)     |✅  | ✅ | ✅  | Shanghai AI Lab |
| Class-to-Image | [dit](https://github.com/mindspore-lab/mindone/blob/master/examples/dit)     | ✅  | ✅  | ✅  | Meta |
| Text-to-Image | [t2i-adapter](https://github.com/mindspore-lab/mindone/blob/master/examples/t2i_adapter)     | ✅  | ✅  | ✅  | Shanghai AI Lab |
| Text-to-Image | [ip adapter](https://github.com/mindspore-lab/mindone/blob/master/examples/ip_adapter)     | ✅  | ✅  | ✅  | Tencent  |
| Text-to-3D | [mvdream](https://github.com/mindspore-lab/mindone/blob/master/examples/mvdream) |   ✅ |   ✅ |   ✅ | ByteDance  |
| Image-to-3D | [instantmesh](https://github.com/mindspore-lab/mindone/blob/master/examples/instantmesh) | ✅  | ✅  | ✅  | Tencent  |
| Image-to-3D | [sv3d](https://github.com/mindspore-lab/mindone/blob/master/examples/sv3d) |   ✅ |   ✅ |   ✅ | Stability AI  |
| Text/Image-to-3D | [hunyuan3d-1.0](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuan3d_1)     | ✅ | ✅ | ✅ | Tencent |

### supported captioner
| task | model  | inference | finetune | pretrain | features  |
| :---   |  :---   |  :---:    |  :---:  |  :---:     |  :--  |
| Image-Text-to-Text | [pllava](https://github.com/mindspore-lab/mindone/tree/master/tools/captioners/PLLaVA) 🔥|  ✅ |  ✖️ |  ✖️   | support video and image captioning |
