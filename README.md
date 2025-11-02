# MindSpore ONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation.

ONE is short for "ONE for all"

## News
- [2025.10.29] We release [v0.4.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.4.0). This release brings **200+ new state-of-the-art AI models** across language, vision, and audio domains. Test it now!
- [2025.04.10] We release [v0.3.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.3.0). More than 15 SoTA generative models are added, including Flux, CogView4, OpenSora2.0, Movie Gen 30B , CogVideoX 5B~30B. Have fun!
- [2025.02.21] We support DeepSeek [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-7B), a SoTA multimodal understanding and generation model. See [here](examples/janus)
- [2024.11.06] [v0.2.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.2.0) is released

## Quick tour

To install v0.4.0, please install [MindSpore 2.6.0/2.7.0](https://www.mindspore.cn/install) and run `pip install mindone`

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
| [pipeline](https://github.com/mindspore-lab/mindone/tree/v0.4.0/mindone/diffusers/pipelines) | support text-to-image,text-to-video,text-to-audio tasks 160+
| [models](https://github.com/mindspore-lab/mindone/tree/v0.4.0/mindone/diffusers/models) | support audoencoder & transformers base models same as hf diffusers 50+
| [schedulers](https://github.com/mindspore-lab/mindone/tree/v0.4.0/mindone/diffusers/schedulers) | support diffusion schedulers (e.g., ddpm and dpm solver) same as hf diffusers 35+

### supported models under mindone/examples

| model  | inference | finetune | pretrain | organization  |
|  :---   |  :---:    |  :---:  |  :---:     |  :--  |
| [hunyuanvideo-i2v](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/hunyuanvideo-i2v)  |  ✅  | ✖️  | ✖️  | Tencent |
| [wan2.1](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/wan2_1)  |  ✅  |  ✖️  |  ✖️   | Alibaba  |
| [wan2.2](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/wan2_2)  |  ✅  |  ✖️  |  ✖️   | Alibaba  |
| [cogview4](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/cogview)  | ✅ | ✖️  | ✖️  | Zhipuai |
| [step_video_t2v](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/step_video_t2v)  | ✅   | ✖️  | ✖️   | StepFun  |
| [qwen2_vl](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/transformers/qwen2_vl) |  ✅ |  ✖️ |  ✖️   | Alibaba |
| [lang_sam](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/lang_sam) | ✅ | ✖️ | ✖️ | Luca Medeiros |
| [sam2](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/sam2) | ✅ | ✖️ | ✖️ | Meta |
| [sparktts](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/sparktts) | ✅ | ✖️ | ✖️ | SparkAudio |
| [flux](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/diffusers/dreambooth)   | ✅ | ✅ | ✖️  | Black Forest Lab |
| [janus](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/janus)   | ✅  | ✅  | ✅  | DeepSeek |
| [emu3](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/emu3)   | ✅  | ✅  | ✅  |  BAAI |
| [var](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/var) | ✅  | ✅  | ✅  | ByteDance  |
| [hpcai open sora 1.2/2.0](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/opensora_hpcai)      | ✅ | ✅ | ✅ | HPC-AI Tech  |
| [cogvideox 1.5 5B~30B ](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/diffusers/cogvideox_factory)  | ✅ |  ✅  | ✅  | Zhipu  |
| [open sora plan 1.3](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/opensora_pku)  | ✅ | ✅ | ✅ | PKU |
| [hunyuanvideo](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/hunyuanvideo) | ✅  | ✅  | ✅  | Tencent  |
| [movie gen 30B](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/moviegen)   | ✅ | ✅ | ✅ | Meta |
| [canny_edit](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/canny_edit) | ✅ | ✅ | ✅ | VayneXie |
| [mmada](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/mmada) | ✅ | ✅ | ✅ | Gen-Verse |
| [omnigen](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/omnigen) | ✅ | ✅ | ✅ | VectorVision |
| [omnigen2](https://github.com/mindspore-lab/mindone/blob/v0.4.0/examples/omnigen2) | ✅ | ✅ | ✅ | VectorVision |
