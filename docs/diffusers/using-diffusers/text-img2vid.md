 <!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Video generation

Video generation models include a temporal dimension to bring images, or frames, together to create a video. These models are trained on large-scale datasets of high-quality text-video pairs to learn how to combine the modalities to ensure the generated video is coherent and realistic.

## Popular models

### CogVideoX

[CogVideoX](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) uses a 3D causal Variational Autoencoder (VAE) to compress videos along the spatial and temporal dimensions, and it includes a stack of expert transformer blocks with a 3D full attention mechanism to better capture visual, semantic, and motion information in the data.

The CogVideoX family also includes models capable of generating videos from images and videos in addition to text. The image-to-video models are indicated by **I2V** in the checkpoint name, and they should be used with the [`CogVideoXImageToVideoPipeline`]. The regular checkpoints support video-to-video through the [`CogVideoXVideoToVideoPipeline`].

The example below demonstrates how to generate a video from an image and text prompt with [THUDM/CogVideoX-5b-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V).

```py
import mindspore as ms
import numpy as np
from mindone.diffusers import CogVideoXImageToVideoPipeline
from mindone.diffusers.utils import export_to_video, load_image

prompt = "A vast, shimmering ocean flows gracefully under a twilight sky, its waves undulating in a mesmerizing dance of blues and greens. The surface glints with the last rays of the setting sun, casting golden highlights that ripple across the water. Seagulls soar above, their cries blending with the gentle roar of the waves. The horizon stretches infinitely, where the ocean meets the sky in a seamless blend of hues. Close-ups reveal the intricate patterns of the waves, capturing the fluidity and dynamic beauty of the sea in motion."
image = load_image(image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cogvideox/cogvideox_rocket.png")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    mindspore_dtype=ms.bfloat16
)

video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=np.random.Generator(np.random.PCG64(seed=42)),
)[0][0]
export_to_video(video, "output.mp4", fps=8)
```

### HunyuanVideo

[HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) features a dual-stream to single-stream diffusion transformer (DiT) for learning video and text tokens separately, and then subsequently concatenating the video and text tokens to combine their information. A single multimodal large language model (MLLM) serves as the text encoder, and videos are also spatio-temporally compressed with a 3D causal VAE.
```py
import mindspore as ms
from mindone.diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from mindone.diffusers.utils import export_to_video

transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo", subfolder="transformer", mindspore_dtype=ms.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(
  "hunyuanvideo-community/HunyuanVideo", transformer=transformer, mindspore_dtype=ms.float16
)

video = pipe(
    prompt="A cat walks on the grass, realistic",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
)[0][0]
export_to_video(video, "output.mp4", fps=15)
```

### LTX-Video

[LTX-Video (LTXV)](https://huggingface.co/Lightricks/LTX-Video) is a diffusion transformer (DiT) with a focus on speed. It generates 768x512 resolution videos at 24 frames per second (fps), enabling near real-time generation of high-quality videos.

```py
import mindspore as ms
from mindone.diffusers import LTXPipeline
from mindone.diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", mindspore_dtype=ms.bfloat16)

prompt = "A man walks towards a window, looks out, and then turns around. He has short, dark hair, dark skin, and is wearing a brown coat over a red and gray scarf. He walks from left to right towards a window, his gaze fixed on something outside. The camera follows him from behind at a medium distance. The room is brightly lit, with white walls and a large window covered by a white curtain. As he approaches the window, he turns his head slightly to the left, then back to the right. He then turns his entire body to the right, facing the window. The camera remains stationary as he stands in front of the window. The scene is captured in real-life footage."
video = pipe(
    prompt=prompt,
    width=704,
    height=480,
    num_frames=161,
    num_inference_steps=50,
)[0][0]
export_to_video(video, "output.mp4", fps=24)
```

### Mochi-1

[Mochi-1](https://huggingface.co/genmo/mochi-1-preview) introduces the Asymmetric Diffusion Transformer (AsymmDiT) and Asymmetric Variational Autoencoder (AsymmVAE) to reduces memory requirements. AsymmVAE causally compresses videos 128x to improve memory efficiency, and AsymmDiT jointly attends to the compressed video tokens and user text tokens. This model is noted for generating videos with high-quality motion dynamics and strong prompt adherence.

```py
import mindspore as ms
from mindone.diffusers import MochiPipeline
from mindone.diffusers.utils import export_to_video

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", variant="bf16", mindspore_dtype=ms.bfloat16)

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
video = pipe(prompt, num_frames=84)[0][0]
export_to_video(video, "output.mp4", fps=30)
```

### StableVideoDiffusion

[StableVideoDiffusion (SVD)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) is based on the Stable Diffusion 2.1 model and it is trained on images, then low-resolution videos, and finally a smaller dataset of high-resolution videos. This model generates a short 2-4 second video from an initial image.

```py
import mindspore as ms
from mindone.diffusers import StableVideoDiffusionPipeline
from mindone.diffusers.utils import load_image, export_to_video
import numpy as np

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", mindspore_dtype=ms.float16, variant="fp16"
)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = np.random.Generator(np.random.PCG64(42))
frames = pipeline(image, decode_chunk_size=8, generator=generator, num_frames=5)[0]
export_to_video(frames, "generated.mp4", fps=7)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/335d58c3-939f-4580-b6ba-0de135b5a918"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated video</figcaption>
  </div>
</div>

### AnimateDiff

[AnimateDiff](https://huggingface.co/guoyww/animatediff) is an adapter model that inserts a motion module into a pretrained diffusion model to animate an image. The adapter is trained on video clips to learn motion which is used to condition the generation process to create a video. It is faster and easier to only train the adapter and it can be loaded into most diffusion models, effectively turning them into “video models”.

Load a `MotionAdapter` and pass it to the [`AnimateDiffPipeline`].

```py
import mindspore as ms
from mindone.diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif
import numpy as np

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", mindspore_dtype=ms.float16)
```

Then load a finetuned Stable Diffusion model with the [`AnimateDiffPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/animatediff/#mindone.diffusers.AnimateDiffPipeline).

```py
pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, mindspore_dtype=ms.float16)
scheduler = DDIMScheduler.from_pretrained(
    "emilianJR/epiCRealism",
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipeline.scheduler = scheduler
```

Create a prompt and generate the video.

```py
output = pipeline(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=50,
    generator=np.random.Generator(np.random.PCG64(49)),
)
frames = output[0][0]
export_to_gif(frames, "animation.gif")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/4011ad22-98a0-40e3-99e7-cb43a4d59174"/>
</div>

## Configure model parameters

There are a few important parameters you can configure in the pipeline that'll affect the video generation process and quality. Let's take a closer look at what these parameters do and how changing them affects the output.

### Number of frames

The `num_frames` parameter determines how many video frames are generated per second. A frame is an image that is played in a sequence of other frames to create motion or a video. This affects video length because the pipeline generates a certain number of frames per second (check a pipeline's API reference for the default value). To increase the video duration, you'll need to increase the `num_frames` parameter.

```py
import mindspore as ms
import numpy as np
from mindone.diffusers import StableVideoDiffusionPipeline
from mindone.diffusers.utils import load_image, export_to_video

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", mindspore_dtype=ms.float16, variant="fp16"
)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = np.random.Generator(np.random.PCG64(42))
frames = pipeline(image, decode_chunk_size=8, generator=generator, num_frames=25)[0][0]
export_to_video(frames, "generated.mp4", fps=7)

### Guidance scale

The `guidance_scale` parameter controls how closely aligned the generated video and text prompt or initial image is. A higher `guidance_scale` value means your generated video is more aligned with the text prompt or initial image, while a lower `guidance_scale` value means your generated video is less aligned which could give the model more "creativity" to interpret the conditioning input.

!!! tip

    SVD uses the `min_guidance_scale` and `max_guidance_scale` parameters for applying guidance to the first and last frames respectively.

```py
import mindspore as ms
from mindone.diffusers import I2VGenXLPipeline
from mindone.diffusers.utils import export_to_gif, load_image
import numpy as np

pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", mindspore_dtype=ms.float16, variant="fp16")

image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
image = load_image(image_url).convert("RGB")
image = image.resize((image.width // 2, image.height // 2))

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = np.random.Generator(np.random.PCG64(0))

frames = pipeline(
    prompt=prompt,
    image=image,
    height=image.height,
    width=image.width,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=1.0,
    generator=generator
)[0][0]
export_to_gif(frames, "i2v.gif")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c056d473-0886-4b01-8b6a-121392e692af"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale=9.0</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/e98132c4-afae-4154-975c-2678d9f3b956"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale=1.0</figcaption>
  </div>
</div>

### Negative prompt

A negative prompt deters the model from generating things you don’t want it to. This parameter is commonly used to improve overall generation quality by removing poor or bad features such as “low resolution” or “bad details”.

```py
import mindspore as ms
from mindone.diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from mindone.diffusers.utils import export_to_gif
import numpy as np

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", mindspore_dtype=ms.float16)

pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, mindspore_dtype=ms.float16)
scheduler = DDIMScheduler.from_pretrained(
    "emilianJR/epiCRealism",
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipeline.scheduler = scheduler

output = pipeline(
    prompt="360 camera shot of a sushi roll in a restaurant",
    negative_prompt="Distorted, discontinuous, ugly, blurry, low resolution, motionless, static",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=50,
    generator=np.random.Generator(np.random.PCG64(0)),
)
frames = output[0][0]
export_to_gif(frames, "animation.gif")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/704d21aa-c65b-411a-82a2-e483f3333f5c"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">no negative prompt</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/70c74d5d-fb2f-4dba-9401-1d41a80c55e5"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">negative prompt applied</figcaption>
  </div>
</div>

### Model-specific parameters

There are some pipeline parameters that are unique to each model such as adjusting the motion in a video or adding noise to the initial image.

Stable Video Diffusion provides additional micro-conditioning for the frame rate with the `fps` parameter and for motion with the `motion_bucket_id` parameter. Together, these parameters allow for adjusting the amount of motion in the generated video.

There is also a `noise_aug_strength` parameter that increases the amount of noise added to the initial image. Varying this parameter affects how similar the generated video and initial image are. A higher `noise_aug_strength` also increases the amount of motion. To learn more, read the [Micro-conditioning](../using-diffusers/svd.md#micro-conditioning) guide.
