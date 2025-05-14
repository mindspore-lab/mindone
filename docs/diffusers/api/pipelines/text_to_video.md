<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

!!! warning

	🧪 This pipeline is for research purposes only.

# Text-to-video

[ModelScope Text-to-Video Technical Report](https://arxiv.org/abs/2308.06571) is by Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, Shiwei Zhang.

The abstract from the paper is:

*This paper introduces ModelScopeT2V, a text-to-video synthesis model that evolves from a text-to-image synthesis model (i.e., Stable Diffusion). ModelScopeT2V incorporates spatio-temporal blocks to ensure consistent frame generation and smooth movement transitions. The model could adapt to varying frame numbers during training and inference, rendering it suitable for both image-text and video-text datasets. ModelScopeT2V brings together three components (i.e., VQGAN, a text encoder, and a denoising UNet), totally comprising 1.7 billion parameters, in which 0.5 billion parameters are dedicated to temporal capabilities. The model demonstrates superior performance over state-of-the-art methods across three evaluation metrics. The code and an online demo are available at https://modelscope.cn/models/damo/text-to-video-synthesis/summary.*

You can find additional information about Text-to-Video on the [project page](https://modelscope.cn/models/damo/text-to-video-synthesis/summary), [original codebase](https://github.com/modelscope/modelscope/), and try it out in a [demo](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis). Official checkpoints can be found at [damo-vilab](https://huggingface.co/damo-vilab) and [cerspense](https://huggingface.co/cerspense).

## Usage example

### `text-to-video-ms-1.7b`

Let's start by generating a short video with the default length of 16 frames (2s at 8 fps):

```python
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", mindspore_dtype=ms.float16, variant="fp16")

prompt = "Spiderman is surfing"
video_frames = pipe(prompt)[0][0]
video_path = export_to_video(video_frames)
video_path
```

Diffusers supports different optimization techniques to improve the latency
and memory footprint of a pipeline. Since videos are often more memory-heavy than images,
we can enable VAE slicing to keep the memory footprint at bay.

Let's generate a video of 8 seconds (64 frames) on the same NPU using VAE slicing:

```python
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", mindspore_dtype=ms.float16, variant="fp16")

# memory optimization
pipe.enable_vae_slicing()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=64)[0][0]
video_path = export_to_video(video_frames)
video_path
```

We can also use a different scheduler easily, using the same method we'd use for Stable Diffusion:

```python
import mindspore as ms
from mindone.diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from mindone.diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", mindspore_dtype=ms.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25)[0][0]
video_path = export_to_video(video_frames)
video_path
```

Here are some sample outputs:

<table>
    <tr>
        <td><center>
        An astronaut riding a horse.
        <br>
        <img src="https://github.com/user-attachments/assets/f8665e8d-f1b7-499f-a430-631848cd9b77"
            alt="An astronaut riding a horse."
            style="width: 300px;" />
        </center></td>
        <td ><center>
        Darth vader surfing in waves.
        <br>
        <img src="https://github.com/user-attachments/assets/917323c8-c48a-441c-a68d-b73671949718"
            alt="Darth vader surfing in waves."
            style="width: 300px;" />
        </center></td>
    </tr>
</table>

### `cerspense/zeroscope_v2_576w` & `cerspense/zeroscope_v2_XL`

Zeroscope are watermark-free model and have been trained on specific sizes such as `576x320` and `1024x576`.
One should first generate a video using the lower resolution checkpoint [`cerspense/zeroscope_v2_576w`](https://huggingface.co/cerspense/zeroscope_v2_576w) with [`TextToVideoSDPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/text_to_video/#mindone.diffusers.TextToVideoSDPipeline),
which can then be upscaled using [`VideoToVideoSDPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/text_to_video/#mindone.diffusers.VideoToVideoSDPipeline) and [`cerspense/zeroscope_v2_XL`](https://huggingface.co/cerspense/zeroscope_v2_XL).


```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from mindone.diffusers.utils import export_to_video
from PIL import Image
import numpy as np

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", mindspore_dtype=ms.float16, revision="refs/pr/46")

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=24)[0][0]
video_path = export_to_video(video_frames)
video_path
```

Now the video can be upscaled:

```py
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", mindspore_dtype=ms.float16, revision="refs/pr/34")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

video_frames = pipe(prompt, video=video, strength=0.6)[0][0]
video_path = export_to_video(video_frames)
video_path
```

Here are some sample outputs:

<table>
    <tr>
        <td ><center>
        Darth vader surfing in waves.
        <br>
        <img src="https://github.com/user-attachments/assets/958a8a43-d5ba-41ec-992f-9009e8b05375"
            alt="Darth vader surfing in waves."
            style="width: 576px;" />
        </center></td>
    </tr>
</table>

## Tips

Video generation is memory-intensive and one way to reduce your memory usage is to set `enable_forward_chunking` on the pipeline's UNet so you don't run the entire feedforward layer at once. Breaking it up into chunks in a loop is more efficient.

Check out the [Text or image-to-video](../../using-diffusers/text-img2vid.md) guide for more details about how certain parameters can affect video generation and how to optimize inference by reducing memory usage.

!!! tip

	Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality.

::: mindone.diffusers.TextToVideoSDPipeline

::: mindone.diffusers.VideoToVideoSDPipeline

::: mindone.diffusers.pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput
