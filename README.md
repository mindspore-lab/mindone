# MindONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation.

ONE is short for "ONE for all"

## News
- [2025.04.10] We release MindONE [v0.3.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.3.0). More than 15 SoTA generative models are added, including Flux, CogView4, OpenSora2.0, Movie Gen 30B , CogVideoX 5B~30B. Have fun!
- [2025.02.21] We support DeepSeek [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-7B), a SoTA multimodal understanding and generation model. See [here](examples/janus)
- [2024.11.06] MindONE [v0.2.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.2.0) is released

## Quick tour

To install MindONE v0.3.0, please install [MindSpore 2.5.0](https://www.mindspore.cn/install) and run `pip install mindone`

Alternatively, to install the latest version from the `master` branch, please run.
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
```

We support state-of-the-art diffusion models for generating images, audio, and video. Let's get started using [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) as an example.

**Hello MindSpore** from **Stable Diffusion 3**!

<div>
<img src="https://github.com/townwish4git/mindone/assets/143256262/8c25ae9a-67b1-436f-abf6-eca36738cd17" alt="sd3" width="512" height="512">
</div>

```py
import mindspore
from mindone.diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    mindspore_dtype=mindspore.float16,
)
prompt = "A cat holding a sign that says 'Hello MindSpore'"
image = pipe(prompt)[0][0]
image.save("sd3.png")
```
###  run hf diffusers on mindspore
 - mindone diffusers is under active development, most tasks were tested with mindspore 2.5.0 on Ascend Atlas 800T A2 machines.
 - compatibale with hf diffusers 0.32.2

| component  |  features  
| :---   |  :--  
| [pipeline](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines) | support text-to-image,text-to-video,text-to-audio tasks 160+
| [models](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/models) | support audoencoder & transformers base models same as hf diffusers 50+
| [schedulers](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/schedulers) | support diffusion schedulers (e.g., ddpm and dpm solver) same as hf diffusers 35+

### supported models under mindone/examples

| task | model  | inference | finetune | pretrain | institute  |
| :---   |  :---   |  :---:    |  :---:  |  :---:     |  :--  |
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
| Text-to-Video | [animate diff](https://github.com/mindspore-lab/mindone/blob/master/examples/animatediff) | ✅  | ✅  | ✅  | CUHK |
| Text/Image-to-Video | [video composer](https://github.com/mindspore-lab/mindone/tree/master/examples/videocomposer)     | ✅  | ✅  | ✅  | Alibaba |
| Text-to-Image | [flux](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_flux.md)  🔥 | ✅ | ✅ | ✖️  | Black Forest Lab |
| Text-to-Image | [stable diffusion 3](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_sd3.md) 🔥| ✅ | ✅ | ✖️ | Stability AI |
| Text-to-Image | [kohya_sd_scripts](https://github.com/mindspore-lab/mindone/blob/master/examples/kohya_sd_scripts) | ✅ | ✅ | ✖️  | kohya |
| Text-to-Image | [stable diffusion xl](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/README_sdxl.md)  | ✅ | ✅ | ✅ | Stability AI|
| Text-to-Image | [hunyuan_dit](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuan_dit)     | ✅ | ✅ | ✅ | Tencent |
| Text-to-Image | [pixart_sigma](https://github.com/mindspore-lab/mindone/blob/master/examples/pixart_sigma)     | ✅ | ✅ | ✅ | Huawei |
| Text-to-Image | [fit](https://github.com/mindspore-lab/mindone/blob/master/examples/fit) | ✅ | ✅ | ✅ | Shanghai AI Lab  |
| Class-to-Video | [latte](https://github.com/mindspore-lab/mindone/blob/master/examples/latte)     |✅  | ✅ | ✅  | Shanghai AI Lab |
| Class-to-Image | [dit](https://github.com/mindspore-lab/mindone/blob/master/examples/dit)     | ✅  | ✅  | ✅  | Meta |
| Text-to-3D | [mvdream](https://github.com/mindspore-lab/mindone/blob/master/examples/mvdream) |   ✅ |   ✅ |   ✅ | ByteDance  |
| Image-to-3D | [instantmesh](https://github.com/mindspore-lab/mindone/blob/master/examples/instantmesh) | ✅  | ✅  | ✅  | Tencent  |
| Image-to-3D | [sv3d](https://github.com/mindspore-lab/mindone/blob/master/examples/sv3d) |   ✅ |   ✅ |   ✅ | Stability AI  |
| Text/Image-to-3D | [hunyuan3d-1.0](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuan3d_1)     | ✅ | ✅ | ✅ | Tencent |

### supported captioner
| task | model  | inference | finetune | pretrain | features  |
| :---   |  :---   |  :---:    |  :---:  |  :---:     |  :--  |
| Image-Text-to-Text | [pllava](https://github.com/mindspore-lab/mindone/tree/master/tools/captioners/PLLaVA) 🔥|  ✅ |  ✖️ |  ✖️   | support video and image captioning |
