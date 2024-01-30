# ControlNet based on Stable Diffusion XL
> [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)

## Introduction
ControlNet controls pretrained large diffusion models to support additional input conditions. The ControlNet learns task-specific conditions in an end-to-end way, and the learning is robust even when the training dataset is small. Large diffusion models like Stable Diffusion can be augmented with ControlNets to enable conditional inputs like canny edge maps, segmentation maps, keypoints, etc.

<p align="center">
   <img src="https://github.com/Gaohan123/mindone/assets/20148503/c5c27f00-3c20-479c-a540-70a0c8db0d48" width=500 />
</p>
<p align="center">
  <em> Fig 1. Illustration of a ControlNet [<a href="#references">1</a>] </em>
</p>


## Dependency

- AI framework: MindSpore >= 2.2 
- Hardware: Ascend 910*

```shell
cd example/stable_diffusion_xl
pip install -r requirement.txt
```

## Inferece

### Prepare model weights

1. Convert trained weight from Diffusers, please refer to [here](tools/controlnet_conversion/README.md);

2. Or train your ControlNet using MindONE (coming soon).

### Prepare control signals

Stable Diffusion XL with ControlNet can generate images following the input control signal (e.g. canny edge). You can either prepare (1) a raw image [[Fig 2]()] to be extracted control signal from, or (2) the control signal image itself [[Fig 3]()].




### Generate image

```shell
python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base_controlnet.yaml \
  --weight checkpoints/sd_xl_base_1.0_controlnet_canny_ms.ckpt \
  --guidance_scale 9.0 \
  --device_target Ascend \
  --controlnet_mode canny \
  --image_path /PATH TO/dog2.png \
  --prompt "cute dog, best quality, extremely detailed"   \
  --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"  \
  # --control_path /PATH TO/dog2_canny_edge.png  \
```

The above script is provided in [`script/run_infer_base_controlnet.sh`](script/run_infer_base_controlnet.sh).

Key arguments:
- `weight`: path to the model weights, refer to [Prepare model weights](#prepare-model-weights) chapter.
- `guidance_scale`: the guidance scale for txt2img and img2img tasks. For NoDynamicThresholding, uncond + guidance_scale * (uncond - cond). Note that this scale could heavily impact the inference result.
- `controlnet_mode`: Control mode for controlnet, supported mode: "canny".
- `image_path`: a raw image to be extracted control signal from
- `control_path`: the control signal image itself
- `prompt`: positve text prompt for image generation
- `negative_prompt`: negative text prompt for image generation


⚠️ **If `--control_path` (like [Fig 3]()) is not None, it will be used as control signal, while `--image_path` (like [Fig 2]()) is not in effect.**

You can check all arguments description by running `python demo/sampling_without_streamlit.py -h`.


### Results



## Reference
[1] [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/pdf/2302.05543.pdf)
