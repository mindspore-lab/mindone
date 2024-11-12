<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Video Diffusion

[Stable Video Diffusion (SVD)](https://huggingface.co/papers/2311.15127) is a powerful image-to-video generation model that can generate 2-4 second high resolution (576x1024) videos conditioned on an input image.

This guide will show you how to use SVD to generate short videos from images.

Before you begin, make sure you have the following libraries installed:

```py
!pip install mindone transformers
```

The are two variants of this model, [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and [SVD-XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt). The SVD checkpoint is trained to generate 14 frames and the SVD-XT checkpoint is further finetuned to generate 25 frames.

You'll use the SVD-XT checkpoint for this guide.

!!! warning

    Due to precision issues, modifications are required to ensure StableVideoDiffusionPipeline functions properly. For further details, please refer to the [Limitation](../limitations.md).

```python
import mindspore as ms

from mindone.diffusers import StableVideoDiffusionPipeline
from mindone.diffusers.utils import load_image, export_to_video
import numpy as np

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", mindspore_dtype=ms.float16, variant="fp16"
)

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = np.random.Generator(np.random.PCG64(42))
frames = pipe(image, num_frames=5, decode_chunk_size=8, generator=generator)[0]

export_to_video(frames, "generated.mp4", fps=7)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"source image of a rocket"</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/f15fbb8f-7b4b-4ad1-b66a-9b4cae86628a"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">"generated video from source image"</figcaption>
  </div>
</div>

## Reduce memory usage

Video generation is very memory intensive because you're essentially generating `num_frames` all at once, similar to text-to-image generation with a high batch size. To reduce the memory requirement, there are multiple options that trade-off inference speed for lower memory requirement:

- enable feed-forward chunking: the feed-forward layer runs in a loop instead of running a single feed-forward with a huge batch size.
- reduce `decode_chunk_size`: the VAE decodes frames in chunks instead of decoding them all together. Setting `decode_chunk_size=1` decodes one frame at a time and uses the least amount of memory (we recommend adjusting this value based on your NPU memory) but the video might have some flickering.

```diff
- frames = pipe(image, num_frames=5, decode_chunk_size=8, generator=generator)[0][0]
+ pipe.unet.enable_forward_chunking()
+ frames = pipe(image, num_frames=5, decode_chunk_size=2, generator=generator)[0][0]
```

Using all these tricks together should lower the memory requirement.

## Micro-conditioning

Stable Diffusion Video also accepts micro-conditioning, in addition to the conditioning image, which allows more control over the generated video:

- `fps`: the frames per second of the generated video.
- `motion_bucket_id`: the motion bucket id to use for the generated video. This can be used to control the motion of the generated video. Increasing the motion bucket id increases the motion of the generated video.
- `noise_aug_strength`: the amount of noise added to the conditioning image. The higher the values the less the video resembles the conditioning image. Increasing this value also increases the motion of the generated video.

For example, to generate a video with more motion, use the `motion_bucket_id` and `noise_aug_strength` micro-conditioning parameters:

```python
import mindspore as ms

from mindone.diffusers import StableVideoDiffusionPipeline
from mindone.diffusers.utils import load_image, export_to_video
import numpy as np

pipe = StableVideoDiffusionPipeline.from_pretrained(
  "stabilityai/stable-video-diffusion-img2vid-xt", mindspore_dtype=ms.float16, variant="fp16"
)

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = np.random.Generator(np.random.PCG64(7))
frames = pipe(image, num_frames=5, decode_chunk_size=8, generator=generator, motion_bucket_id=180, noise_aug_strength=0.1)[0]
export_to_video(frames, "generated.mp4", fps=7)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/ac776105-bbc3-40eb-9d9b-bc0e3965a721"/>
</div>
