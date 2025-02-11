<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# T2I-Adapter

[T2I-Adapter](https://hf.co/papers/2302.08453) is a lightweight adapter for controlling and providing more accurate
structure guidance for text-to-image models. It works by learning an alignment between the internal knowledge of the
text-to-image model and an external control signal, such as edge detection or depth estimation.

The T2I-Adapter design is simple, the condition is passed to four feature extraction blocks and three downsample
blocks. This makes it fast and easy to train different adapters for different conditions which can be plugged into the
text-to-image model. T2I-Adapter is similar to [ControlNet](controlnet.md) except it is smaller (~77M parameters) and
faster because it only runs once during the diffusion process. The downside is that performance may be slightly worse
than ControlNet.

This guide will show you how to use T2I-Adapter with different Stable Diffusion models and how you can compose multiple
T2I-Adapters to impose more than one condition.

!!! tip

    There are several T2I-Adapters available for different conditions, such as color palette, depth, sketch, pose, and
    segmentation. Check out the [TencentARC](https://hf.co/TencentARC) repository to try them out!

Before you begin, make sure you have the following libraries installed.

```py
# uncomment to install the necessary libraries in Colab
#!pip install mindone
```

## Text-to-image

Text-to-image models rely on a prompt to generate an image, but sometimes, text alone may not be enough to provide more
accurate structural guidance. T2I-Adapter allows you to provide an additional control image to guide the generation
process. For example, you can provide a canny image (a white outline of an image on a black background) to guide the
model to generate an image with a similar structure.

=== "Stable Diffusion 1.5"

    Create a canny image with the [opencv-library](https://github.com/opencv/opencv-python).

    ```py
    import cv2
    import numpy as np
    from PIL import Image
    from mindone.diffusers.utils import load_image

    image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = Image.fromarray(image)
    ```

    Now load a T2I-Adapter conditioned on [canny images](https://hf.co/TencentARC/t2iadapter_canny_sd15v2) and pass it to
    the [`StableDiffusionAdapterPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/adapter/#mindone.diffusers.StableDiffusionAdapterPipeline).

    ```py
    import mindspore as ms
    from mindone.diffusers import StableDiffusionAdapterPipeline, T2IAdapter
    import numpy as np

    adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_canny_sd15v2", mindspore_dtype=ms.float16)
    pipeline = StableDiffusionAdapterPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        adapter=adapter,
        mindspore_dtype=ms.float16,
    )
    ```

    Finally, pass your prompt and control image to the pipeline.

    ```py
    generator = np.random.Generator(np.random.PCG64(0))

    image = pipeline(
        prompt="cinematic photo of a plush and soft midcentury style rug on a wooden floor, 35mm photograph, film, professional, 4k, highly detailed",
        image=image,
        generator=generator,
    )[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
      <img class="rounded-xl" src="https://github.com/user-attachments/assets/6c672914-51d1-49e1-a28e-2e033dbfeac0"/>
    </div>

=== "Stable Diffusion XL"

    !!! warning

        ⚠️ MindONE currently does not support the full process for the following code, as MindONE does not yet support `CannyDetector` from controlnet_aux.canny. Therefore, you need to prepare the `canny image` in advance to continue the process.

    Load a canny image.

    ```py
    from diffusers.utils import load_image

    image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
    image = load_image("path/to/canny_image")
    ```

    Now load a T2I-Adapter conditioned on [canny images](https://hf.co/TencentARC/t2i-adapter-canny-sdxl-1.0) and pass it
    to the [`StableDiffusionXLAdapterPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/adapter/#mindone.diffusers.StableDiffusionXLAdapterPipeline).

    ```py
    import mindspore as ms
    from mindone.diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
    import numpy as np

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms.float16)
    adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-canny-sdxl-1.0", mindspore_dtype=ms.float16)
    pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        adapter=adapter,
        vae=vae,
        scheduler=scheduler,
        mindspore_dtype=ms.float16,
    )
    ```

    Finally, pass your prompt and control image to the pipeline.

    ```py
    generator = np.random.Generator(np.random.PCG64(0))

    image = pipeline(
      prompt="cinematic photo of a plush and soft midcentury style rug on a wooden floor, 35mm photograph, film, professional, 4k, highly detailed",
      image=image,
      generator=generator,
    )[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
      <img class="rounded-xl" src="https://github.com/user-attachments/assets/2c3c7ebc-7206-49f0-89d8-a507122c42f6"/>
    </div>

## MultiAdapter

T2I-Adapters are also composable, allowing you to use more than one adapter to impose multiple control conditions on an
image. For example, you can use a pose map to provide structural control and a depth map for depth control. This is
enabled by the [`MultiAdapter`] class.

Let's condition a text-to-image model with a pose and depth adapter. Create and place your depth and pose image and in a list.

```py
from mindone.diffusers.utils import load_image

pose_image = load_image(
    "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_input.png"
)
depth_image = load_image(
    "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_input.png"
)
cond = [pose_image, depth_image]
prompt = ["Santa Claus walking into an office room with a beautiful city view"]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/depth_sample_input.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">depth image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/diffusers/docs-images/resolve/main/t2i-adapter/keypose_sample_input.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">pose image</figcaption>
  </div>
</div>

Load the corresponding pose and depth adapters as a list in the [`MultiAdapter`] class.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionAdapterPipeline, MultiAdapter, T2IAdapter

adapters = MultiAdapter(
    [
        T2IAdapter.from_pretrained("TencentARC/t2iadapter_keypose_sd14v1"),
        T2IAdapter.from_pretrained("TencentARC/t2iadapter_depth_sd14v1"),
    ]
)
adapters = adapters.to(ms.float16)
```

Finally, load a [`StableDiffusionAdapterPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/adapter/#mindone.diffusers.StableDiffusionAdapterPipeline) with the adapters, and pass your prompt and conditioned images to
it. Use the [`adapter_conditioning_scale`] to adjust the weight of each adapter on the image.

```py
pipeline = StableDiffusionAdapterPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    mindspore_dtype=ms.float16,
    adapter=adapters,
)

image = pipeline(prompt, cond, adapter_conditioning_scale=[0.7, 0.7])[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <img class="rounded-xl" src="https://github.com/user-attachments/assets/1bd8d71b-0b48-4e20-b8ea-dd7d75054eb6"/>
</div>
