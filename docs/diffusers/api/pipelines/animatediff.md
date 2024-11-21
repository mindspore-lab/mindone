<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text-to-Video Generation with AnimateDiff

## Overview

[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725) by Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai.

The abstract of the paper is the following:

*With the advance of text-to-image models (e.g., Stable Diffusion) and corresponding personalization techniques such as DreamBooth and LoRA, everyone can manifest their imagination into high-quality images at an affordable cost. Subsequently, there is a great demand for image animation techniques to further combine generated static images with motion dynamics. In this report, we propose a practical framework to animate most of the existing personalized text-to-image models once and for all, saving efforts in model-specific tuning. At the core of the proposed framework is to insert a newly initialized motion modeling module into the frozen text-to-image model and train it on video clips to distill reasonable motion priors. Once trained, by simply injecting this motion modeling module, all personalized versions derived from the same base T2I readily become text-driven models that produce diverse and personalized animated images. We conduct our evaluation on several public representative personalized text-to-image models across anime pictures and realistic photographs, and demonstrate that our proposed framework helps these models generate temporally smooth animation clips while preserving the domain and diversity of their outputs. Code and pre-trained weights will be publicly available at [this https URL](https://animatediff.github.io/).*

## Available Pipelines

| Pipeline                                                                                                                                                                           | Tasks  
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| [AnimateDiffPipeline](https://github.com/mindspore-lab/mindone/blob/master/mindone/diffusers/pipelines/animatediff/pipeline_animatediff.py)                                        | *Text-to-Video Generation with AnimateDiff* |
| [AnimateDiffSDXLPipeline](https://github.com/mindspore-lab/mindone/blob/master/mindone/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py)                               | *Video-to-Video Generation with AnimateDiff* |
[AnimateDiffVideoToVideoPipeline](https://github.com/mindspore-lab/mindone/blob/master/mindone/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py)                | *Video-to-Video Generation with AnimateDiff* |

## Available checkpoints

Motion Adapter checkpoints can be found under [guoyww](https://huggingface.co/guoyww/). These checkpoints are meant to work with any model based on Stable Diffusion 1.4/1.5.

## Usage example

### AnimateDiffPipeline

AnimateDiff works with a MotionAdapter checkpoint and a Stable Diffusion model checkpoint. The MotionAdapter is a collection of Motion Modules that are responsible for adding coherent motion across image frames. These modules are applied after the Resnet and Attention blocks in Stable Diffusion UNet.

The following example demonstrates how to use a *MotionAdapter* checkpoint with Diffusers for inference based on StableDiffusion-1.4/1.5.

```python
import mindspore as ms
from mindone.diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", mindspore_dtype=ms.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, mindspore_dtype=ms.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
)
frames = output[0][0]
export_to_gif(frames, "animation.gif")
```

Here are some sample outputs:

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://github.com/user-attachments/assets/eb0e9b32-24b6-45be-8c7a-58656cdcf83c"
            alt="masterpiece, bestquality, sunset"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

!!! tip

    AnimateDiff tends to work better with finetuned Stable Diffusion models. If you plan on using a scheduler that can clip samples, make sure to disable it by setting `clip_sample=False` in the scheduler as this can also have an adverse effect on generated samples. Additionally, the AnimateDiff checkpoints can be sensitive to the beta schedule of the scheduler. We recommend setting this to `linear`.


### AnimateDiffSDXLPipeline

AnimateDiff can also be used with SDXL models. This is currently an experimental feature as only a beta release of the motion adapter checkpoint is available.

```python
import mindspore as ms
from mindone.diffusers.models import MotionAdapter
from mindone.diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
from mindone.diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", mindspore_dtype=ms.float16)
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)

pipe = AnimateDiffSDXLPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    scheduler=scheduler,
    mindspore_dtype=ms.float16,
    variant="fp16",
)


output = pipe(
    prompt="a panda surfing in the ocean, realistic, high quality",
    negative_prompt="low quality, worst quality",
    num_inference_steps=20,
    guidance_scale=8,
    width=1024,
    height=1024,
    num_frames=16,
)

frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

### AnimateDiffVideoToVideoPipeline

AnimateDiff can also be used to generate visually similar videos or enable style/character/background or other edits starting from an initial video, allowing you to seamlessly explore creative possibilities.

```python
import imageio
import requests
import numpy as np
import mindspore as ms
from mindone.diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif
from io import BytesIO
from PIL import Image

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", mindspore_dtype=ms.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(model_id, motion_adapter=adapter, mindspore_dtype=ms.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# helper function to load videos
def load_video(file_path: str):
    images = []

    if file_path.startswith(('http://', 'https://')):
        # If the file_path is a URL
        response = requests.get(file_path)
        response.raise_for_status()
        content = BytesIO(response.content)
        vid = imageio.get_reader(content)
    else:
        # Assuming it's a local file path
        vid = imageio.get_reader(file_path)

    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)

    return images

video = load_video("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif")

output = pipe(
    video = video,
    prompt="panda playing a guitar, on a boat, in the ocean, high quality",
    negative_prompt="bad quality, worse quality",
    guidance_scale=7.5,
    num_inference_steps=25,
    strength=0.5,
    generator=np.random.Generator(np.random.PCG64(42)),
)
frames = output[0][0]
export_to_gif(frames, "animation.gif")
```

Here are some sample outputs:

<table>
    <tr>
      <th align=center>Source Video</th>
      <th align=center>Output Video</th>
    </tr>
    <tr>
        <td align=center>
          raccoon playing a guitar
          <br />
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif"
              alt="racoon playing a guitar"
              style="width: 300px;" />
        </td>
        <td align=center>
          panda playing a guitar
          <br/>
          <img src="https://github.com/user-attachments/assets/79cbc306-afa6-48a8-9cbe-8bf171a6de09"
              alt="panda playing a guitar"
              style="width: 300px;" />
        </td>
    </tr>
    <tr>
        <td align=center>
          closeup of margot robbie, fireworks in the background, high quality
          <br />
          <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-2.gif"
              alt="closeup of margot robbie, fireworks in the background, high quality"
              style="width: 300px;" />
        </td>
        <td align=center>
          closeup of tony stark, robert downey jr, fireworks
          <br/>
          <img src="https://github.com/user-attachments/assets/f732fa6a-454c-4cd9-b730-ff3682c3fadc"
              alt="closeup of tony stark, robert downey jr, fireworks"
              style="width: 300px;" />
        </td>
    </tr>
</table>


## Using Motion LoRAs

Motion LoRAs are a collection of LoRAs that work with the `guoyww/animatediff-motion-adapter-v1-5-2` checkpoint. These LoRAs are responsible for adding specific types of motion to the animations.

```python
import numpy as np
import mindspore as ms
from mindone.diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", mindspore_dtype=ms.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, mindspore_dtype=ms.float16)
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out"
)

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    beta_schedule="linear",
    timestep_spacing="linspace",
    steps_offset=1,
)
pipe.scheduler = scheduler

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=np.random.Generator(np.random.PCG64(42)),
)
frames = output[0][0]
export_to_gif(frames, "animation.gif")
```

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://github.com/user-attachments/assets/b2e9f729-fd9a-4c7f-a6c5-a1ef14de317d"
            alt="masterpiece, bestquality, sunset"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

```python
import numpy as np
import mindspore as ms
from mindone.diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", mindspore_dtype=ms.float16)

# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, mindspore_dtype=ms.float16)

pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out",
)

pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-pan-left", adapter_name="pan-left",
)
pipe.set_adapters(["zoom-out", "pan-left"], adapter_weights=[1.0, 1.0])

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)

pipe.scheduler = scheduler

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=np.random.Generator(np.random.PCG64(42)),
)

frames = output[0][0]

export_to_gif(frames, "animation.gif")

```

<table>
    <tr>
        <td><center>
        masterpiece, bestquality, sunset.
        <br>
        <img src="https://github.com/user-attachments/assets/1faba057-aa83-4b84-81d2-a2fcb1d05f9a"
            alt="masterpiece, bestquality, sunset"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

## Using AnimateLCM

[AnimateLCM](https://animatelcm.github.io/) is a motion module checkpoint and an [LCM LoRA](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm_lora) that have been created using a consistency learning strategy that decouples the distillation of the image generation priors and the motion generation priors.

```python
import numpy as np
import mindspore as ms
from mindone.diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")

output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=1.5,
    num_inference_steps=6,
    generator=np.random.Generator(np.random.PCG64(0)),
)
frames = output[0][0]
export_to_gif(frames, "animatelcm.gif")
```

<table>
    <tr>
        <td><center>
        A space rocket, 4K.
        <br>
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatelcm-output.gif"
            alt="A space rocket, 4K"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

AnimateLCM is also compatible with existing [Motion LoRAs](https://huggingface.co/collections/dn6/animatediff-motion-loras-654cb8ad732b9e3cf4d3c17e).

```python
import numpy as np
import mindspore as ms
from mindone.diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-up", adapter_name="tilt-up")

pipe.set_adapters(["lcm-lora", "tilt-up"], [1.0, 0.8])

output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=1.5,
    num_inference_steps=6,
    generator=np.random.Generator(np.random.PCG64(0)),
)
frames = output[0][0]
export_to_gif(frames, "animatelcm-motion-lora.gif")
```

<table>
    <tr>
        <td><center>
        A space rocket, 4K.
        <br>
        <img src="https://github.com/user-attachments/assets/71454cea-979c-4244-845a-48ad4873dc7c"
            alt="A space rocket, 4K"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

::: mindone.diffusers.pipelines.AnimateDiffPipeline

::: mindone.diffusers.pipelines.AnimateDiffSDXLPipeline

::: mindone.diffusers.pipelines.AnimateDiffVideoToVideoPipeline

::: mindone.diffusers.pipelines.animatediff.AnimateDiffPipelineOutput
