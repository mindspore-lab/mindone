<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Image-to-Video Generation with PIA (Personalized Image Animator)

## Overview

[PIA: Your Personalized Image Animator via Plug-and-Play Modules in Text-to-Image Models](https://arxiv.org/abs/2312.13964) by Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, Kai Chen

Recent advancements in personalized text-to-image (T2I) models have revolutionized content creation, empowering non-experts to generate stunning images with unique styles. While promising, adding realistic motions into these personalized images by text poses significant challenges in preserving distinct styles, high-fidelity details, and achieving motion controllability by text. In this paper, we present PIA, a Personalized Image Animator that excels in aligning with condition images, achieving motion controllability by text, and the compatibility with various personalized T2I models without specific tuning. To achieve these goals, PIA builds upon a base T2I model with well-trained temporal alignment layers, allowing for the seamless transformation of any personalized T2I model into an image animation model. A key component of PIA is the introduction of the condition module, which utilizes the condition frame and inter-frame affinity as input to transfer appearance information guided by the affinity hint for individual frame synthesis in the latent space. This design mitigates the challenges of appearance-related image alignment within and allows for a stronger focus on aligning with motion-related guidance.

[Project page](https://pi-animator.github.io/)

## Available Pipelines

|                                                    Pipeline                                                     |                 Tasks                 | Demo |
|:---------------------------------------------------------------------------------------------------------------:|:-------------------------------------:|:----:|
|  [PIAPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pia/pipeline_pia.py)  | *Image-to-Video Generation with PIA*  |      |

## Available checkpoints

Motion Adapter checkpoints for PIA can be found under the [OpenMMLab org](https://huggingface.co/openmmlab/PIA-condition-adapter). These checkpoints are meant to work with any model based on Stable Diffusion 1.5

## Usage example

PIA works with a MotionAdapter checkpoint and a Stable Diffusion 1.5 model checkpoint. The MotionAdapter is a collection of Motion Modules that are responsible for adding coherent motion across image frames. These modules are applied after the Resnet and Attention blocks in the Stable Diffusion UNet. In addition to the motion modules, PIA also replaces the input convolution layer of the SD 1.5 UNet model with a 9 channel input convolution layer.

The following example demonstrates how to use PIA to generate a video from a single image.

```python
import mindspore as ms
from mindone.diffusers import (
    EulerDiscreteScheduler,
    MotionAdapter,
    PIAPipeline,
)
from mindone.diffusers.utils import export_to_gif, load_image
import numpy as np

adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter", mindspore_dtype=ms.float16)
pipe = PIAPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    motion_adapter=adapter,
    mindspore_dtype=ms.float16,
    revision="refs/pr/8",
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing()

image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
)
image = image.resize((512, 512))
prompt = "cat in a field"
negative_prompt = "wrong white balance, dark, sketches,worst quality,low quality"

generator = np.random.Generator(np.random.PCG64(seed=0))
output = pipe(image=image, prompt=prompt, generator=generator)
frames = output[0][0]
export_to_gif(frames, "pia-animation.gif")
```

Here are some sample outputs:

<table>
    <tr>
        <td><center>
        oringinal picture.
        <br>
        <img src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png"
            alt="oringinal picture"
            style="width: 300px;" />
        </center></td>
        <td><center>
        cat in a field.
        <br>
        <img src="https://github.com/user-attachments/assets/46299910-3b61-468c-975f-43090eea2fea"
            alt="cat in a field"
            style="width: 300px;" />
        </center></td>
    </tr>
</table>


!!! tip

    If you plan on using a scheduler that can clip samples, make sure to disable it by setting `clip_sample=False` in the scheduler as this can also have an adverse effect on generated samples. Additionally, the PIA checkpoints can be sensitive to the beta schedule of the scheduler. We recommend setting this to `linear`.

::: mindone.diffusers.PIAPipeline

::: mindone.diffusers.pipelines.pia.PIAPipelineOutput
