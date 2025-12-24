# MindSpore ONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation.

ONE is short for "ONE for all"

## News

- [2025.12.24] We release [v0.5.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.5.0), compatibility with 🤗 Transformers v4.57.1 ([70+ new models](./mindone/transformers/SUPPORT_LIST.md)) and 🤗 Diffusers v0.35.2, plus previews of v0.36 pipelines like Flux2, QwenImageEditPlus, Lucy and Kandinsky5. Also introduces initial ComfyUI integration. Happy exploring!
- [2025.11.02] [v0.4.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.4.0) is released, with 280+ transformers models and 70+ diffusers pipelines supported. See [here](https://github.com/mindspore-lab/mindone/blob/refs/tags/v0.4.0/CHANGELOG.md)
- [2025.04.10] We release [v0.3.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.3.0). More than 15 SoTA generative models are added, including Flux, CogView4, OpenSora2.0, Movie Gen 30B, CogVideoX 5B~30B. Have fun!
- [2025.02.21] We support DeepSeek [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-7B), a SoTA multimodal understanding and generation model. See [here](examples/janus)
- [2024.11.06] [v0.2.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.2.0) is released

## Quick tour

To install v0.5.0, please install [MindSpore 2.6.0 - 2.7.1](https://www.mindspore.cn/install) and run `pip install mindone`

Alternatively, to install the latest version from the `master` branch, please run:
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
 - mindone diffusers is under active development, most tasks were tested with MindSpore 2.6.0-2.7.1 on Ascend Atlas 800T A2 machines
 - compatible with 🤗 diffusers v0.35.2, preview supports for SoTA v0.36 pipelines, see [support list](./mindone/diffusers/SUPPORT_LIST.md)
 - 18+ [training examples](./examples/diffusers) - controlnet, dreambooth, lora and more


###  run hf transformers on mindspore
 - mindone transformers is under active development, most tasks were tested with mindspore 2.6.0-2.7.1 on Ascend Atlas 800T A2 machines
 - compatibale with 🤗 transformers v4.57.1
 - providing 350+ state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal model for inference, see [support list](./mindone/transformers/SUPPORT_LIST.md)

### supported models under mindone/examples

| task | model  | inference | finetune | pretrain | institute  |
| :---   |  :---   |  :---:    |  :---:  |  :---:     |  :--  |
| Text/Image-to-Video | [wan2.1](https://github.com/mindspore-lab/mindone/blob/master/examples/wan2_1) 🔥 |  ✅  |  ✖️  |  ✖️   | Alibaba  |
| Text/Image-to-Video | [wan2.2](https://github.com/mindspore-lab/mindone/blob/master/examples/wan2_2) 🔥🔥 |  ✅  |  ✅  |  ✖️   | Alibaba  |
| Audio/Image-Text-to-Text | [qwen2_5_omni](https://github.com/mindspore-lab/mindone/blob/master/examples/transformers/qwen2_5_omni) 🔥🔥|  ✅ |  ✅ |  ✖️   | Alibaba |
| Image/Video-Text-to-Text  | [qwen2_5_vl](https://github.com/mindspore-lab/mindone/tree/master/examples/transformers/qwen2_5_vl) 🔥🔥|  ✅ | ✅  |  ✖️   | Alibaba |
| Any-to-Any  | [qwen3_omni_moe](https://github.com/mindspore-lab/mindone/tree/master/examples/transformers/qwen3_omni_moe) 🔥🔥🔥 |  ✅ | ✖️   |  ✖️   | Alibaba |
| Image-Text-to-Text | [qwen3_vl/qwen3_vl_moe](https://github.com/mindspore-lab/mindone/tree/master/examples/transformers/qwen3_vl) 🔥🔥🔥 |  ✅ | ✖️   |  ✖️   | Alibaba |
| Text-to-Image | [qwen_image](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers/qwenimage) 🔥🔥🔥 |  ✅ | ✅   |  ✖️   | Alibaba |
| Text-to-Text | [minicpm](https://github.com/mindspore-lab/mindone/tree/master/examples/transformers/minicpm) 🔥🔥 | ✅ | ✖️   |  ✖️   | OpenBMB |
| Any-to-Any | [janus](https://github.com/mindspore-lab/mindone/blob/master/examples/janus)   | ✅  | ✅  | ✅  | DeepSeek |
| Any-to-Any | [emu3](https://github.com/mindspore-lab/mindone/blob/master/examples/emu3)   | ✅  | ✅  | ✅  |  BAAI |
| Class-to-Image | [var](https://github.com/mindspore-lab/mindone/blob/master/examples/var) | ✅  | ✅  | ✅  | ByteDance  |
| Text-to-Image | [omnigen2](https://github.com/mindspore-lab/mindone/blob/master/examples/omnigen2) 🔥 | ✅ | ✅  | ✖️  | VectorSpaceLab |
| Text/Image-to-Video | [hpcai open sora 1.2/2.0](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_hpcai)    | ✅ | ✅ | ✅ | HPC-AI Tech  |
| Text/Image-to-Video | [cogvideox 1.5 5B~30B ](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/cogvideox_factory)  | ✅ |  ✅  | ✅  | Zhipu  |
| Image/Text-to-Text | [glm4v](https://github.com/mindspore-lab/mindone/tree/master/examples/transformers/glm4v) 🔥 | ✅ | ✖️   |  ✖️ | Zhipu |
| Text-to-Video | [open sora plan 1.3](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_pku)  | ✅ | ✅ | ✅ | PKU |
| Text-to-Video | [hunyuanvideo](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo) | ✅  | ✅  | ✅  | Tencent  |
| Image-to-Video | [hunyuanvideo-i2v](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo-i2v) 🔥 |  ✅  | ✖️  | ✖️  | Tencent |
| Text-to-Video | [movie gen 30B](https://github.com/mindspore-lab/mindone/blob/master/examples/moviegen)   | ✅ | ✅ | ✅ | Meta |
| Segmentation | [lang_sam](https://github.com/mindspore-lab/mindone/tree/master/examples/lang_sam) 🔥 | ✅ | ✖️ | ✖️ | Meta |
| Segmentation | [sam2](https://github.com/mindspore-lab/mindone/tree/master/examples/sam2) |✅  | ✖️ |✖️  | Meta |
| Text-to-Video | [step_video_t2v](https://github.com/mindspore-lab/mindone/blob/master/examples/step_video_t2v) | ✅   | ✖️  | ✖️   | StepFun  |
| Text-to-Speech | [sparktts](https://github.com/mindspore-lab/mindone/tree/master/examples/sparktts) |✅   | ✖️  | ✖️   | Spark Audio |
| Text-to-Image | [flux](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_flux.md)  | ✅ | ✅ | ✖️  | Black Forest Lab |
| Text-to-Image | [stable diffusion 3](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_sd3.md) | ✅ | ✅ | ✖️ | Stability AI |


### supported captioner
| task | model  | inference | finetune | pretrain | features  |
| :---   |  :---   |  :---:    |  :---:  |  :---:     |  :--  |
| Image-Text-to-Text | [pllava](https://github.com/mindspore-lab/mindone/tree/master/tools/captioners/PLLaVA) |  ✅ |  ✖️ |  ✖️   | support video and image captioning |

### training-free acceleration
Introduce [dit infer acceleration](https://github.com/mindspore-lab/mindone/blob/master/examples/accelerated_dit_pipelines/README.md) - DiTCache, PromptGate and FBCache with Taylorseer, tested on sd3 and flux.1.
