<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Text or image-to-video

Driven by the success of text-to-image diffusion models, generative video models are able to generate short clips of video from a text prompt or an initial image. These models extend a pretrained diffusion model to generate videos by adding some type of temporal and/or spatial convolution layer to the architecture. A mixed dataset of images and videos are used to train the model which learns to output a series of video frames based on the text or image conditioning.

This guide will show you how to generate videos, how to configure video model parameters, and how to control video generation.

## Popular models

!!! tip

    Discover other cool and trending video generation models on the Hub [here](https://huggingface.co/models?pipeline_tag=text-to-video&sort=trending)!

[Stable Video Diffusions (SVD)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid), [I2VGen-XL](https://huggingface.co/ali-vilab/i2vgen-xl/) and [AnimateDiff](https://huggingface.co/guoyww/animatediff) are popular models used for video diffusion. Each model is distinct. For example, AnimateDiff inserts a motion modeling module into a frozen text-to-image model to generate personalized animated images, whereas SVD is entirely pretrained from scratch with a three-stage training process to generate short high-quality videos.

### Stable Video Diffusion

[SVD](../api/pipelines/svd.md) is based on the Stable Diffusion 2.1 model and it is trained on images, then low-resolution videos, and finally a smaller dataset of high-resolution videos. This model generates a short 2-4 second video from an initial image. You can learn more details about model, like micro-conditioning, in the [Stable Video Diffusion](../using-diffusers/svd.md) guide.

Begin by loading the [`StableVideoDiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/svd/#mindone.diffusers.StableVideoDiffusionPipeline) and passing an initial image to generate a video from.

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

### I2VGen-XL

[I2VGen-XL](../api/pipelines/i2vgenxl.md) is a diffusion model that can generate higher resolution videos than SVD and it is also capable of accepting text prompts in addition to images. The model is trained with two hierarchical encoders (detail and global encoder) to better capture low and high-level details in images. These learned details are used to train a video diffusion model which refines the video resolution and details in the generated video.

You can use I2VGen-XL by loading the [`I2VGenXLPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/i2vgenxl/#mindone.diffusers.I2VGenXLPipeline), and passing a text and image prompt to generate a video.

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
generator = np.random.Generator(np.random.PCG64(8888))

frames = pipeline(
    prompt=prompt,
    image=image,
    height=image.height,
    width=image.width,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9.0,
    generator=generator
)[0][0]
export_to_gif(frames, "i2v.gif")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/5a624adc-e734-4fb5-b077-e6e4fc672dd9"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/0f8ad1d0-889b-49df-96db-e9a6d8ab4c26"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated video</figcaption>
  </div>
</div>

### AnimateDiff

[AnimateDiff](../api/pipelines/animatediff.md) is an adapter model that inserts a motion module into a pretrained diffusion model to animate an image. The adapter is trained on video clips to learn motion which is used to condition the generation process to create a video. It is faster and easier to only train the adapter and it can be loaded into most diffusion models, effectively turning them into "video models".

Start by loading a [`MotionAdapter`].

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
from mindone.diffusers import I2VGenXLPipeline
from mindone.diffusers.utils import export_to_gif, load_image
import numpy as np

pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", mindspore_dtype=ms.float16, variant="fp16")

image_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/i2vgen_xl_images/img_0009.png"
image = load_image(image_url).convert("RGB")
image = image.resize((image.width // 2, image.height // 2))

prompt = "Papers were floating in the air on a table in the library"
negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
generator = np.random.Generator(np.random.PCG64(8888))

frames = pipeline(
    prompt=prompt,
    image=image,
    height=image.height,
    width=image.width,
    num_inference_steps=50,
    negative_prompt=negative_prompt,
    guidance_scale=9.0,
    generator=generator,
    num_frames=25,
)[0][0]
export_to_gif(frames, "i2v.gif")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/ff750130-a292-4163-973b-30582fcab398"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">num_frames=14</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/e93a46d8-7161-494e-807e-8dfd1a213605"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">num_frames=25</figcaption>
  </div>
</div>

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
