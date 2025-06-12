<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# Wan

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

[Wan 2.1](https://github.com/Wan-Video/Wan2.1) by the Alibaba Wan Team.

<!-- TODO(aryan): update abstract once paper is out -->

## Generating Videos with Wan 2.1

We will first need to install some addtional dependencies.

```shell
pip install -u ftfy imageio-ffmpeg imageio
```

### Text to Video Generation

The following example requires 11GB VRAM to run and uses the smaller `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` model. You can switch it out
for the larger `Wan2.1-I2V-14B-720P-Diffusers` or `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` if you have at least 35GB VRAM available.

```python
from mindone.diffusers import WanPipeline
from mindone.diffusers.utils import export_to_video
import mindspore as ms

# Available models: Wan-AI/Wan2.1-I2V-14B-720P-Diffusers or Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

pipe = WanPipeline.from_pretrained(model_id, mindspore_dtype=ms.bfloat16)

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames)[0][0]
export_to_video(frames, "wan-t2v.mp4", fps=16)
```

!!! tip

    You can improve the quality of the generated video by running the decoding step in full precision.

```python
from mindone.diffusers import WanPipeline, AutoencoderKLWan
from mindone.diffusers.utils import export_to_video
import mindspore as ms

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", mindspore_dtype=ms.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, mindspore_dtype=ms.bfloat16)

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
num_frames = 33

frames = pipe(prompt=prompt, num_frames=num_frames)[0][0]
export_to_video(frames, "wan-t2v.mp4", fps=16)
```

### Image to Video Generation

The Image to Video pipeline requires loading the `AutoencoderKLWan` and the `CLIPVisionModel` components in full precision. The following example will need at least
35GB of VRAM to run.

```python
import mindspore as ms
import numpy as np
from mindone.diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from mindone.diffusers.utils import export_to_video, load_image
from mindone.transformers import CLIPVisionModel

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", mindspore_dtype=ms.float32
)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", mindspore_dtype=ms.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, mindspore_dtype=ms.bfloat16
)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

num_frames = 33

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=5.0,
)[0][0]
export_to_video(output, "wan-i2v.mp4", fps=16)
```

### Video to Video Generation

```python
import mindspore as ms
from mindone.diffusers.utils import load_video, export_to_video
from mindone.diffusers import AutoencoderKLWan, WanVideoToVideoPipeline, UniPCMultistepScheduler

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", mindspore_dtype=ms.float32
)
pipe = WanVideoToVideoPipeline.from_pretrained(
    model_id, vae=vae, mindspore_dtype=ms.bfloat16
)
flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=flow_shift
)

prompt = "A robot standing on a mountain top. The sun is setting in the background"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
)
output = pipe(
    video=video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=512,
    guidance_scale=7.0,
    strength=0.7,
)[0][0]

export_to_video(output, "wan-v2v.mp4", fps=16)
```

## Using Single File Loading with Wan 2.1

The `WanTransformer3DModel` and `AutoencoderKLWan` models support loading checkpoints in their original format via the `from_single_file` loading
method.

```python
import mindspore as ms
from mindone.diffusers import WanPipeline, WanTransformer3DModel

ckpt_path = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"
transformer = WanTransformer3DModel.from_single_file(ckpt_path, mindspore_dtype=ms.bfloat16)

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", transformer=transformer)
```

## Recommendations for Inference
- Keep `AutencoderKLWan` in `torch.float32` for better decoding quality.
- `num_frames` should satisfy the following constraint: `(num_frames - 1) % 4 == 0`
- For smaller resolution videos, try lower values of `shift` (between `2.0` to `5.0`) in the [Scheduler](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/flow_match_euler_discrete/#mindone.diffusers.FlowMatchEulerDiscreteScheduler.shift). For larger resolution videos, try higher values (between `7.0` and `12.0`). The default value is `3.0` for Wan.

::: mindone.diffusers.WanPipeline

::: mindone.diffusers.WanImageToVideoPipeline

::: mindone.diffusers.WanVideoToVideoPipeline

::: mindone.diffusers.pipelines.wan.pipeline_output.WanPipelineOutput
