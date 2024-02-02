# ControlNet based on Stable Diffusion XL
> [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

## Introduction
ControlNet controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

<p align="center">
   <img src="https://github.com/Gaohan123/mindone/assets/20148503/c5c27f00-3c20-479c-a540-70a0c8db0d48" width=700 />
</p>
<p align="center">
  <em> Fig 1. Illustration of a ControlNet [<a href="#reference">1</a>] </em>
</p>


## Dependency

- AI framework: MindSpore >= 2.2
- Hardware: Ascend 910*

```shell
cd examples/stable_diffusion_xl
pip install -r requirement.txt
```

## Inferece

### Prepare model weights

1. Convert trained weight from Diffusers, please refer to [here](tools/controlnet_conversion/README.md);

2. Or train your ControlNet using MindONE (coming soon).

### Prepare control signals

Stable Diffusion XL with ControlNet can generate images following the input control signal (e.g. canny edge). You can either prepare (1) a raw image (Fig 2) to be extracted control signal from, or (2) the control signal image itself (Fig 3).

<div align="center">
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/f8a7ef86-3d4a-4d07-b99e-46156c356e73" width=350 />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/7eaff4e2-d9a4-44e6-a059-e8e1074f2301" width=350 />
</div>
<p align="center">
<em> Fig 2. raw image to be extracted control signal </em>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<em> Fig 3. control signal image (canny edge) </em>
</p>


### Generate images

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base_controlnet.yaml \
  --weight checkpoints/sd_xl_base_1.0_controlnet_canny_ms.ckpt \
  --guidance_scale 9.0 \
  --device_target Ascend \
  --controlnet_mode canny \
  --prompt "cute dog, best quality, extremely detailed"   \
  --image_path /PATH TO/dog2.png \
  # --control_path /PATH TO/dog2_canny_edge.png  \
```

The above script is provided in [`script/run_infer_base_controlnet.sh`](scripts/run_infer_base_controlnet.sh).

Key arguments:
- `weight`: path to the model weights, refer to [Prepare model weights](#prepare-model-weights) chapter.
- `guidance_scale`: the guidance scale for txt2img and img2img tasks. For NoDynamicThresholding, uncond + guidance_scale * (uncond - cond). Note that this scale could heavily impact the inference result.
- `controlnet_mode`: Control mode for controlnet, supported mode: "canny".
- `image_path`: a raw image to be extracted control signal from.
- `control_path`: the control signal image itself.
- `prompt`: positve text prompt for image generation.


⚠️ **If `--control_path` (like Fig 3) is not None, it will be used as control signal, while `--image_path` (like Fig 2) is not in effect.**

You can check all arguments description by running `python demo/sampling_without_streamlit.py -h`.


### Inference results

<div align="center">
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/f8a7ef86-3d4a-4d07-b99e-46156c356e73" width=30% />
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/7eaff4e2-d9a4-44e6-a059-e8e1074f2301" width=30% />
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/13c84292-5bdb-4048-97aa-683112b04f34" width=30% />
</div>
<p align="center">
<em> Fig 4. From left to right: raw image - extracted canny edge - inference result. </em>
</br>
<em> Prompt: "cute dog, best quality, extremely detailed". </em>
</p>

<div align="center">
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/ecdc18c9-36bf-4d49-b7b2-6216400f1d5a" width=30% />
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/00f0ed2c-c078-4ed3-bee6-cb8c66fb36fd" width=30% />
<img src="https://github.com/HaoyangLee/mindone/assets/20376974/31da3e94-4633-4506-b2eb-cfd65001d3e8" width=30% />
</div>
<p align="center">
<em> Fig 5. From left to right: raw image - extracted canny edge - inference result. </em>
</br>
<em> Prompt: "beautiful bird standing on a trunk, natural color, best quality, extremely detailed". </em>
</p>

## Reference
[1] [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)
