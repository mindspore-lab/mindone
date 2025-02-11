<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ControlNet

ControlNet is a type of model for controlling image diffusion models by conditioning the model with an additional input image. There are many types of conditioning inputs (canny edge, user sketching, human pose, depth, and more) you can use to control a diffusion model. This is hugely useful because it affords you greater control over image generation, making it easier to generate specific images without experimenting with different text prompts or denoising values as much.

!!! tip

    Check out Section 3.5 of the [ControlNet](https://huggingface.co/papers/2302.05543) paper v1 for a list of ControlNet implementations on various conditioning inputs. You can find the official Stable Diffusion ControlNet conditioned models on [lllyasviel](https://huggingface.co/lllyasviel)'s Hub profile, and more [community-trained](https://huggingface.co/models?other=stable-diffusion&other=controlnet) ones on the Hub.

    For Stable Diffusion XL (SDXL) ControlNet models, you can find them on the 🤗 [Diffusers](https://huggingface.co/diffusers) Hub organization, or you can browse [community-trained](https://huggingface.co/models?other=stable-diffusion-xl&other=controlnet) ones on the Hub.

A ControlNet model has two sets of weights (or blocks) connected by a zero-convolution layer:

- a *locked copy* keeps everything a large pretrained diffusion model has learned
- a *trainable copy* is trained on the additional conditioning input

Since the locked copy preserves the pretrained model, training and implementing a ControlNet on a new conditioning input is as fast as finetuning any other model because you aren't training the model from scratch.

This guide will show you how to use ControlNet for text-to-image, image-to-image, inpainting, and more! There are many types of ControlNet conditioning inputs to choose from, but in this guide we'll only focus on several of them. Feel free to experiment with other conditioning inputs!

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries
#!pip install mindone transformers opencv-python
```

## Text-to-image

For text-to-image, you normally pass a text prompt to the model. But with ControlNet, you can specify an additional conditioning input. Let's condition the model with a canny image, a white outline of an image on a black background. This way, the ControlNet can use the canny image as a control to guide the model to generate an image with the same outline.

Load an image and use the [opencv-python](https://github.com/opencv/opencv-python) library to extract the canny image:

```py
from mindone.diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np

original_image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vermeer_canny_edged.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">canny image</figcaption>
  </div>
</div>

Next, load a ControlNet model conditioned on canny edge detection and pass it to the [`StableDiffusionControlNetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet#mindone.diffusers.StableDiffusionControlNetPipeline). Use the faster [`UniPCMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/unipc/#mindone.diffusers.UniPCMultistepScheduler) and enable model offloading to speed up inference and reduce memory usage.

```py
from mindone.diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import mindspore as ms

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", mindspore_dtype=ms.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, mindspore_dtype=ms.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```

Now pass your prompt and canny image to the pipeline:

```py
output = pipe(
    "the mona lisa", image=canny_image
)[0][0]
make_image_grid([original_image, canny_image, output], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <img src="https://github.com/user-attachments/assets/b46737b1-959f-42ce-a83d-a9ec1155706a"/>
</div>

## Image-to-image

!!! warning

    ⚠️ MindONE currently does not support the full process for extracting the depth map, as MindONE does not yet support depth-estimation [~transformers.Pipeline] from mindone.transformers. Therefore, you need to prepare the depth map in advance to continue the process.

For image-to-image, you'd typically pass an initial image and a prompt to the pipeline to generate a new image. With ControlNet, you can pass an additional conditioning input to guide the model. Let's condition the model with a depth map, an image which contains spatial information. This way, the ControlNet can use the depth map as a control to guide the model to generate an image that preserves spatial information.

You'll use the [`StableDiffusionControlNetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet#mindone.diffusers.StableDiffusionControlNetImg2ImgPipeline) for this task, which is different from the [`StableDiffusionControlNetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet#mindone.diffusers.StableDiffusionControlNetPipeline) because it allows you to pass an initial image as the starting point for the image generation process.

You can process and retrieve the depth map you prepared in advance:

```py
import mindspore as ms
import numpy as np

from mindone.diffusers.utils import load_image, make_image_grid

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
)

def make_hint(depth_image):
    image = depth_image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = ms.Tensor.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint

hint = make_hint(depth_image).unsqueeze(0).half()
```

Next, load a ControlNet model conditioned on depth maps and pass it to the [`StableDiffusionControlNetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet#mindone.diffusers.StableDiffusionControlNetImg2ImgPipeline). Use the faster [`UniPCMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/unipc/#mindone.diffusers.UniPCMultistepScheduler) and enable model offloading to speed up inference and reduce memory usage.

```py
from mindone.diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import mindspore as ms

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", mindspore_dtype=ms.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, mindspore_dtype=ms.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```

Now pass your prompt, initial image, and depth map to the pipeline:

```py
output = pipe(
    "lego batman and robin", image=image, control_image=depth_map,
)[0][0]
make_image_grid([image, output], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/eef32a0c-7a25-4d13-8576-7c961583de02"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

## Inpainting

For inpainting, you need an initial image, a mask image, and a prompt describing what to replace the mask with. ControlNet models allow you to add another control image to condition a model with. Let’s condition the model with an inpainting mask. This way, the ControlNet can use the inpainting mask as a control to guide the model to generate an image within the mask area.

Load an initial image and a mask image:

```py
from mindone.diffusers.utils import load_image, make_image_grid

init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg"
)
init_image = init_image.resize((512, 512))

mask_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg"
)
mask_image = mask_image.resize((512, 512))
make_image_grid([init_image, mask_image], rows=1, cols=2)
```

Create a function to prepare the control image from the initial and mask images. This'll create a tensor to mark the pixels in `init_image` as masked if the corresponding pixel in `mask_image` is over a certain threshold.

```py
import numpy as np
import mindspore as ms

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = ms.Tensor.from_numpy(image)
    return image

control_image = make_inpaint_condition(init_image, mask_image)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint.jpg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-inpaint-mask.jpg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">mask image</figcaption>
  </div>
</div>

Load a ControlNet model conditioned on inpainting and pass it to the [`StableDiffusionControlNetInpaintPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet/#mindone.diffusers.StableDiffusionControlNetInpaintPipeline). Use the faster [`UniPCMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/unipc/#mindone.diffusers.UniPCMultistepScheduler) and enable model offloading to speed up inference and reduce memory usage.

```py
from mindone.diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", mindspore_dtype=ms.float16, use_safetensors=True)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, mindspore_dtype=ms.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```

Now pass your prompt, initial image, mask image, and control image to the pipeline:

```py
output = pipe(
    "corgi face with large ears, detailed, pixar, animated, disney",
    num_inference_steps=20,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
)[0][0]
make_image_grid([init_image, mask_image, output], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <img src="https://github.com/user-attachments/assets/f368896b-96a1-4856-82f9-7e9cec177320"/>
</div>

## Guess mode

[Guess mode](https://github.com/lllyasviel/ControlNet/discussions/188) does not require supplying a prompt to a ControlNet at all! This forces the ControlNet encoder to do its best to "guess" the contents of the input control map (depth map, pose estimation, canny edge, etc.).

Guess mode adjusts the scale of the output residuals from a ControlNet by a fixed ratio depending on the block depth. The shallowest `DownBlock` corresponds to 0.1, and as the blocks get deeper, the scale increases exponentially such that the scale of the `MidBlock` output becomes 1.0.

!!! tip

    Guess mode does not have any impact on prompt conditioning and you can still provide a prompt if you want.

Set `guess_mode=True` in the pipeline, and it is [recommended](https://github.com/lllyasviel/ControlNet#guess-mode--non-prompt-mode) to set the `guidance_scale` value between 3.0 and 5.0.

```py
from mindone.diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from mindone.diffusers.utils import load_image, make_image_grid
import numpy as np
import mindspore as ms
from PIL import Image
import cv2

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, use_safetensors=True)

original_image = load_image("https://huggingface.co/takuma104/controlnet_dev/resolve/main/bird_512x512.png")

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

image = pipe("", image=canny_image, guess_mode=True, guidance_scale=3.0)[0][0]
make_image_grid([original_image, canny_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/e9d8bfc6-d803-4767-8c71-848beb8c1001"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">regular mode with prompt</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/8cc0c520-d975-4c51-b76f-b0d488413df8"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guess mode without prompt</figcaption>
  </div>
</div>

## ControlNet with Stable Diffusion XL

There aren't too many ControlNet models compatible with Stable Diffusion XL (SDXL) at the moment, but diffusers have trained two full-sized ControlNet models for SDXL conditioned on canny edge detection and depth maps. We're also experimenting with creating smaller versions of these SDXL-compatible ControlNet models so it is easier to run on resource-constrained hardware. You can find these checkpoints on the [🤗 Diffusers Hub organization](https://huggingface.co/diffusers)!

Let's use a SDXL ControlNet conditioned on canny images to generate an image. Start by loading an image and prepare the canny image:

```py
from mindone.diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from mindone.diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import mindspore as ms

original_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
make_image_grid([original_image, canny_image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hf-logo-canny.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">canny image</figcaption>
  </div>
</div>

Load a SDXL ControlNet model conditioned on canny edge detection and pass it to the [`StableDiffusionXLControlNetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet_sdxl/#mindone.diffusers.StableDiffusionXLControlNetPipeline). You can also enable model offloading to reduce memory usage.

```py
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    mindspore_dtype=ms.float16,
    use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    mindspore_dtype=ms.float16,
    use_safetensors=True
)
```

Now pass your prompt (and optionally a negative prompt if you're using one) and canny image to the pipeline:

!!! tip

    The [`controlnet_conditioning_scale`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet/#mindone.diffusers.StableDiffusionControlNetPipeline.__call__.controlnet_conditioning_scale) parameter determines how much weight to assign to the conditioning inputs. A value of 0.5 is recommended for good generalization, but feel free to experiment with this number!

```py
prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    controlnet_conditioning_scale=0.5,
)[0][0]
make_image_grid([original_image, canny_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/704d0777-d16b-4878-a3c0-42baefe5b965"/>
</div>

You can use [`StableDiffusionXLControlNetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet_sdxl/#mindone.diffusers.StableDiffusionXLControlNetPipeline) in guess mode as well by setting the parameter to `True`:

```py
from mindone.diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from mindone.diffusers.utils import load_image, make_image_grid
import numpy as np
import mindspore as ms
import cv2
from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

original_image = load_image(
    "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, mindspore_dtype=ms.float16, use_safetensors=True
)

image = np.array(original_image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

image = pipe(
    prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=0.5, image=canny_image, guess_mode=True,
)[0][0]
make_image_grid([original_image, canny_image, image], rows=1, cols=3)
```

!!! tip

    You can use a refiner model with `StableDiffusionXLControlNetPipeline` to improve image quality, just like you can with a regular `StableDiffusionXLPipeline`.
    See the [Refine image quality](./sdxl.md#refine-image-quality) section to learn how to use the refiner model.
    Make sure to use `StableDiffusionXLControlNetPipeline` and pass `image` and `controlnet_conditioning_scale`.

    ```py
    base = StableDiffusionXLControlNetPipeline(...)
    image = base(
        prompt=prompt,
        controlnet_conditioning_scale=0.5,
        image=canny_image,
        num_inference_steps=40,
        denoising_end=0.8,
        output_type="latent",
    )[0]
    # rest exactly as with StableDiffusionXLPipeline
    ```

## MultiControlNet

!!! warning

    ⚠️ MindONE currently does not support the full process for human pose estimation, as MindONE does not yet support `OpenposeDetector` from controlnet_aux. Therefore, you need to prepare the `human pose image` in advance to continue the process.

!!! tip

    Replace the SDXL model with a model like [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) to use multiple conditioning inputs with Stable Diffusion models.

You can compose multiple ControlNet conditionings from different image inputs to create a *MultiControlNet*. To get better results, it is often helpful to:

1. mask conditionings such that they don't overlap (for example, mask the area of a canny image where the pose conditioning is located)
2. experiment with the [`controlnet_conditioning_scale`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet/#mindone.diffusers.StableDiffusionControlNetPipeline) parameter to determine how much weight to assign to each conditioning input

In this example, you'll combine a canny image and a human pose estimation image to generate a new image.

Prepare the canny image conditioning:

```py
from mindone.diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import cv2

original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
)
image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)

# zero out middle columns of image where pose will be overlaid
zero_start = image.shape[1] // 4
zero_end = zero_start + image.shape[1] // 2
image[:, zero_start:zero_end] = 0

image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
make_image_grid([original_image, canny_image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/landscape_canny_masked.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">canny image</figcaption>
  </div>
</div>

For human pose estimation, prepare the human pose estimation conditioning:

```py
original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"
)
openpose_image = load_image("path/to/openpose_image")
make_image_grid([original_image, openpose_image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/controlnet/person_pose.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">human pose image</figcaption>
  </div>
</div>

Load a list of ControlNet models that correspond to each conditioning, and pass them to the [`StableDiffusionXLControlNetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet_sdxl/#mindone.diffusers.StableDiffusionXLControlNetPipeline). Use the faster [`UniPCMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/unipc/#mindone.diffusers.UniPCMultistepScheduler) and enable model offloading to reduce memory usage.

```py
from mindone.diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
import mindspore as ms
import numpy as np

controlnets = [
    ControlNetModel.from_pretrained(
        "thibaud/controlnet-openpose-sdxl-1.0", mindspore_dtype=ms.float16
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", mindspore_dtype=ms.float16, use_safetensors=True
    ),
]

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnets, vae=vae, mindspore_dtype=ms.float16, use_safetensors=True
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
```

Now you can pass your prompt (an optional negative prompt if you're using one), canny image, and pose image to the pipeline:

```py
prompt = "a giant standing in a fantasy landscape, best quality"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

generator = np.random.Generator(np.random.PCG64(1))

images = [openpose_image.resize((1024, 1024)), canny_image.resize((1024, 1024))]

images = pipe(
    prompt,
    image=images,
    num_inference_steps=25,
    generator=generator,
    negative_prompt=negative_prompt,
    num_images_per_prompt=3,
    controlnet_conditioning_scale=[1.0, 0.8],
)[0]
make_image_grid([original_image, canny_image, openpose_image,
                images[0].resize((512, 512)), images[1].resize((512, 512)), images[2].resize((512, 512))], rows=2, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
	<img class="rounded-xl" src="https://github.com/user-attachments/assets/b3c9b484-9b93-4596-9dc8-60ac629acfba"/>
</div>
