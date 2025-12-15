# MindONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation.

ONE is short for "ONE for all"

## News
### MVDream is supported!
MVDream is a diffusion model that is able to generate consistent multiview images from a given text prompt. It shows that learning from both 2D and 3D data, a multiview diffusion model can achieve the generalizability of 2D diffusion models and the consistency of 3D renderings. Details can be found [here](examples/mvdream/README.md)
<table class="center" border="0" style="width: 100%; text-align: left;">
<tr>
  <th>Input Prompt</th>
  <th style="text-align: center;">Rendererd MView Video</th>
  <th style="text-align: center;">3D Mesh Generation in Color</th>
</tr>
<tr>
        <td> <code>an astronaut riding a horse</code> </td>
        <td><video src="https://github.com/user-attachments/assets/fb424c82-7262-4591-b182-8e9f4789f6f8" autoplay muted loop playsinline></video></td>
        <td><div class="sketchfab-embed-wrapper"> <iframe title="an astronaut riding a horse_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/2191db5b61834839aac5238f60d70e59/embed"> </iframe> </div></td>
</tr>
        <td> <code>Michelangelo style statue of dog reading news on a cellphone </code> </td>
        <td><video src="https://github.com/user-attachments/assets/80d11282-4a5d-4b5a-9c68-3dc5dd16fb00" autoplay muted loop playsinline></video></td>
        <td><div class="sketchfab-embed-wrapper"> <iframe title="Michelangelo style statue of dog reading news_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/c21773f276884a5db7d47e41926645e4/embed"> </iframe> </div></td>
</table>



These videos are rendered from the trained 3D implicit field in our MVDream model. Color meshes are extracted with the script [`MVDream-threestudio/extract_color_mesh.py`](MVDream-threestudio/extract_color_mesh.py).


### InstantMesh is supported!
We support [instantmesh](https://github.com/TencentARC/InstantMesh) for the 3D mesh generation using the multiview images extracted from [the sv3d pipeline](https://github.com/mindspore-lab/mindone/pull/574).
<p align="center" width="100%">
  <img width="746" alt="Capture" src="https://github.com/user-attachments/assets/be5cf033-8f89-4cad-97dc-2bf76c1b7a4d">
</p>


Using the multiview images input from 3D mesh extracted from [the sv3d pipeline](examples/sv3d/simple_video_sample.py), we extracted 3D meshes as below. Please kindly find the input illustrated by following the link to the sv3d pipeline below.

| <p align="center"> akun </p>                                                                                                                                                                                                                                                                                                                                                                          | <p align="center"> anya </p>                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <div class="sketchfab-embed-wrapper"><iframe title="akun_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/c8b5b475529d48589b85746aab638d2b/embed"></iframe></div> | <div class="sketchfab-embed-wrapper"><iframe title="anya_ms" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" allow="autoplay; fullscreen; xr-spatial-tracking" xr-spatial-tracking execution-while-out-of-viewport execution-while-not-rendered web-share src="https://sketchfab.com/models/180fd247ba2f4437ac665114a4cd4dca/embed"></iframe></div> |

The illustrations here are better viewed in viewers than with HTML support (e.g., the vscode built-in viewer).


### Stable Video 3D is supported!
<p align="center"><img width="600" alt="Output Vis"
src="https://github.com/mindspore-lab/mindone/assets/13991298/0da9cff8-f90a-4fd2-b042-8f92b387a46b"/>
<br><em>Output Multiview Images (21x576x576)</em></br>
</p>

A camera-guided diffusion model that can generate the multiview snippet of a given image! Details can be found [here](examples/sv3d/README.md).
<details>
<summary>More Inference Demos
</summary>

| Input                                                                                                                | Output                     |
|----------------------------------------------------------------------------------------------------------------------|----------------------------|
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/4f7a0c2f-65c1-4d0a-9861-068b811e0701"/><br/>aaa</p>            | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/ad492ad6-0a7a-4227-8809-b3c8ecf4db65"/><br/>aaa multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/64c269c4-dfee-4495-bede-c7841b137895"/><br/>akun</p>           | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/0588fb26-aa1c-44e0-9b85-e001c6b2e67e"/><br/>akun multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/9655bf80-559c-40bb-8953-c8bdea2d11a3"/><br/>anya</p>           | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/95a15c51-6fa7-4587-8e94-4f979270923f"/><br/>anya multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/8bae9feb-17a1-4cbe-ae56-1f719416e3e8"/><br/>bag</p>            | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/5abff1b5-494f-4321-ae27-6125409515b8"/><br/>bag multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/1b6a650a-d203-461c-a60e-fd03e9434ea8"/><br/>groot</p>          | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/413421b0-79d4-48b3-89a8-13958ff2125d"/><br/>groot multiview</p> |
| <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/5458d1db-807b-4b2e-9f0a-22415f2a0f5e"/><br/>princess-large</p> | <p align="center"><img width="300" src="https://github.com/mindspore-lab/mindone/assets/13991298/6bf201a8-da31-4424-8304-42eaf6748501"/><br/>princess-large multiview</p> |

</details>


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
