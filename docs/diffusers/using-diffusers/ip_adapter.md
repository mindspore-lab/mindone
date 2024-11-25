<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# IP-Adapter

[IP-Adapter](https://hf.co/papers/2308.06721) is an image prompt adapter that can be plugged into diffusion models to enable image prompting without any changes to the underlying model. Furthermore, this adapter can be reused with other models finetuned from the same base model and it can be combined with other adapters like [ControlNet](../using-diffusers/controlnet.md). The key idea behind IP-Adapter is the *decoupled cross-attention* mechanism which adds a separate cross-attention layer just for image features instead of using the same cross-attention layer for both text and image features. This allows the model to learn more image-specific features.

!!! tip

    Learn how to load an IP-Adapter in the [Load adapters](../using-diffusers/loading_adapters.md#ip-adapter) guide, and make sure you check out the [IP-Adapter Plus](../using-diffusers/loading_adapters.md#ip-adapter-plus) section which requires manually loading the image encoder.

This guide will walk you through using IP-Adapter for various tasks and use cases.

## General tasks

Let's take a look at how to use IP-Adapter's image prompting capabilities with the [`StableDiffusionXLPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLPipeline) for tasks like text-to-image, image-to-image, and inpainting. We also encourage you to try out other pipelines such as Stable Diffusion, LCM-LoRA, ControlNet, T2I-Adapter, or AnimateDiff!

In all the following examples, you'll see the [`set_ip_adapter_scale`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/ip_adapter/#mindone.diffusers.loaders.ip_adapter.IPAdapterMixin.set_ip_adapter_scale) method. This method controls the amount of text or image conditioning to apply to the model. A value of `1.0` means the model is only conditioned on the image prompt. Lowering this value encourages the model to produce more diverse images, but they may not be as aligned with the image prompt. Typically, a value of `0.5` achieves a good balance between the two prompt types and produces good results.

!!! tip

    In the examples below, try adding `low_cpu_mem_usage=True` to the [`load_ip_adapter`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/ip_adapter/#mindone.diffusers.loaders.ip_adapter.IPAdapterMixin.load_ip_adapter) method to speed up the loading time.

=== "Text-to-image"

    Crafting the precise text prompt to generate the image you want can be difficult because it may not always capture what you'd like to express. Adding an image alongside the text prompt helps the model better understand what it should generate and can lead to more accurate results.

    Load a Stable Diffusion XL (SDXL) model and insert an IP-Adapter into the model with the [`load_ip_adapter`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/ip_adapter/#mindone.diffusers.loaders.ip_adapter.IPAdapterMixin.load_ip_adapter) method. Use the `subfolder` parameter to load the SDXL model weights.

    ```py
    from mindone.diffusers import StableDiffusionXLPipeline
    from mindone.diffusers.utils import load_image
    import mindspore as ms
    import numpy as np

    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16)
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
    pipeline.set_ip_adapter_scale(0.6)
    ```

    Create a text prompt and load an image prompt before passing them to the pipeline to generate an image.

    ```py
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png")
    generator = np.random.Generator(np.random.PCG64(0))
    images = pipeline(
        prompt="a polar bear sitting in a chair drinking a milkshake",
        ip_adapter_image=image,
        negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
        num_inference_steps=100,
        generator=generator,
    )[0]
    images[0]
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
      <div class="flex-1" style="width: 50%">
        <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
      </div>
      <div class="flex-1" style="width: 50%">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/6c8d20fc-9363-4d18-82b8-754b85b8aaa4"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
      </div>
    </div>

=== "Image-to-image"

    IP-Adapter can also help with image-to-image by guiding the model to generate an image that resembles the original image and the image prompt.

    Load a Stable Diffusion XL (SDXL) model and insert an IP-Adapter into the model with the [`load_ip_adapter`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/ip_adapter/#mindone.diffusers.loaders.ip_adapter.IPAdapterMixin.load_ip_adapter) method. Use the `subfolder` parameter to load the SDXL model weights.

    ```py
    from mindone.diffusers import StableDiffusionXLImg2ImgPipeline
    from mindone.diffusers.utils import load_image
    import mindspore as ms
    import numpy as np

    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16)
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
    pipeline.set_ip_adapter_scale(0.6)
    ```

    Pass the original image and the IP-Adapter image prompt to the pipeline to generate an image. Providing a text prompt to the pipeline is optional, but in this example, a text prompt is used to increase image quality.

    ```py
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png").resize((1470, 980))
    ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_2.png").resize((1418, 1890))

    generator = np.random.Generator(np.random.PCG64(4))
    images = pipeline(
        prompt="best quality, high quality",
        image=image,
        ip_adapter_image=ip_image,
        generator=generator,
        strength=0.6,
    )[0]
    images[0]
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
      <div>
        <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
      </div>
      <div>
        <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_2.png"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
      </div>
      <div>
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/6fad4e18-5bc9-4928-a852-37ad38e6c7ef"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
      </div>
    </div>

=== "Inpainting"

    IP-Adapter is also useful for inpainting because the image prompt allows you to be much more specific about what you'd like to generate.

    Load a Stable Diffusion XL (SDXL) model and insert an IP-Adapter into the model with the [`load_ip_adapter`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/ip_adapter/#mindone.diffusers.loaders.ip_adapter.IPAdapterMixin.load_ip_adapter) method. Use the `subfolder` parameter to load the SDXL model weights.

    ```py
    from mindone.diffusers import StableDiffusionXLInpaintPipeline
    from mindone.diffusers.utils import load_image
    import mindspore as ms
    import numpy as np

    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", mindspore_dtype=ms.float16)
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
    pipeline.set_ip_adapter_scale(0.6)
    ```

    Pass a prompt, the original image, mask image, and the IP-Adapter image prompt to the pipeline to generate an image.

    ```py
    mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_mask.png")
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
    ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png")

    generator = np.random.Generator(np.random.PCG64(4))
    images = pipeline(
        prompt="a cute gummy bear waving",
        image=image,
        mask_image=mask_image,
        ip_adapter_image=ip_image,
        generator=generator,
        num_inference_steps=100,
    )[0]
    images[0]
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
      <div>
        <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">original image</figcaption>
      </div>
      <div>
        <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
      </div>
      <div>
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/accaf8a1-5d8e-4072-aae9-67f70880be07"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
      </div>
    </div>

=== "Video"

    IP-Adapter can also help you generate videos that are more aligned with your text prompt. For example, let's load [AnimateDiff](../api/pipelines/animatediff.md) with its motion adapter and insert an IP-Adapter into the model with the [`load_ip_adapter`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/ip_adapter/#mindone.diffusers.loaders.ip_adapter.IPAdapterMixin.load_ip_adapter) method.

    ```py
    import mindspore as ms
    from mindone.diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
    from mindone.diffusers.utils import export_to_gif
    from mindone.diffusers.utils import load_image
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

    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
    ```

    Pass a prompt and an image prompt to the pipeline to generate a short video.

    ```py
    ip_adapter_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png")

    output = pipeline(
        prompt="A cute gummy bear waving",
        negative_prompt="bad quality, worse quality, low resolution",
        ip_adapter_image=ip_adapter_image,
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=50,
        generator=np.random.Generator(np.random.PCG64(0)),
    )
    frames = output[0][0]
    export_to_gif(frames, "gummy_bear.gif")
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
      <div class="flex-1" style="width: 50%">
        <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
      </div>
      <div class="flex-1" style="width: 50%">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/646bdfc2-cc3d-4988-8f62-5aa98a3720b1"/>
        <figcaption class="mt-2 text-center text-sm text-gray-500">generated video</figcaption>
      </div>
    </div>

## Configure parameters

There are a couple of IP-Adapter parameters that are useful to know about and can help you with your image generation tasks. These parameters can make your workflow more efficient or give you more control over image generation.

### Image embeddings

IP-Adapter enabled pipelines provide the `ip_adapter_image_embeds` parameter to accept precomputed image embeddings. This is particularly useful in scenarios where you need to run the IP-Adapter pipeline multiple times because you have more than one image. For example, [multi IP-Adapter](#multi-ip-adapter) is a specific use case where you provide multiple styling images to generate a specific image in a specific style. Loading and encoding multiple images each time you use the pipeline would be inefficient. Instead, you can precompute and save the image embeddings to disk (which can save a lot of space if you're using high-quality images) and load them when you need them.

Call the [`prepare_ip_adapter_image_embeds`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/text2img/#mindone.diffusers.StableDiffusionPipeline.get_guidance_scale_embedding) method to encode and generate the image embeddings. Then you can load the image embeddings by passing them to the `ip_adapter_image_embeds` parameter.

!!! tip

    If you're using IP-Adapter with `ip_adapter_image_embedding` instead of `ip_adapter_image`', you can set `load_ip_adapter(image_encoder_folder=None,...)` because you don't need to load an encoder to generate the image embeddings.

```py
image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=image,
    ip_adapter_image_embeds=None,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

images = pipeline(
    prompt="a polar bear sitting in a chair drinking a milkshake",
    ip_adapter_image_embeds=image_embeds,
    negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
    num_inference_steps=100,
    generator=generator,
)[0]
```

## Specific use cases

IP-Adapter's image prompting and compatibility with other adapters and models makes it a versatile tool for a variety of use cases. This section covers some of the more popular applications of IP-Adapter, and we can't wait to see what you come up with!

### Face model

Generating accurate faces is challenging because they are complex and nuanced. Diffusers supports two IP-Adapter checkpoints specifically trained to generate faces from the [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) repository:

* [ip-adapter-full-face_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-full-face_sd15.safetensors) is conditioned with images of cropped faces and removed backgrounds
* [ip-adapter-plus-face_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-plus-face_sd15.safetensors) uses patch embeddings and is conditioned with images of cropped faces

Additionally, Diffusers supports all IP-Adapter checkpoints trained with face embeddings extracted by `insightface` face models. Supported models are from the [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) repository.

For face models, use the [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter) checkpoint. It is also recommended to use [`DDIMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddim/#mindone.diffusers.DDIMScheduler) or [`EulerDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/euler/#mindone.diffusers.EulerDiscreteScheduler) for face models.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionPipeline, DDIMScheduler
from mindone.diffusers.utils import load_image
import numpy as np

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    mindspore_dtype=ms.float16,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.safetensors")

pipeline.set_ip_adapter_scale(0.5)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")
generator = np.random.Generator(np.random.PCG64(26))

image = pipeline(
    prompt="A photo of Einstein as a chef, wearing an apron, cooking in a French restaurant",
    ip_adapter_image=image,
    negative_prompt="lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=100,
    generator=generator,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/bc1103a0-6159-426a-a07a-07e715c45b25"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

### Multi IP-Adapter

More than one IP-Adapter can be used at the same time to generate specific images in more diverse styles. For example, you can use IP-Adapter-Face to generate consistent faces and characters, and IP-Adapter Plus to generate those faces in a specific style.

!!! tip

    Read the [IP-Adapter Plus](../using-diffusers/loading_adapters.md#ip-adapter-plus) section to learn why you need to manually load the image encoder.

Load the image encoder with [`~transformers.CLIPVisionModelWithProjection`].

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionXLPipeline, DDIMScheduler
from mindone.transformers import CLIPVisionModelWithProjection
from mindone.diffusers.utils import load_image
import numpy as np

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    mindspore_dtype=ms.float16,
)
```

Next, you'll load a base model, scheduler, and the IP-Adapters. The IP-Adapters to use are passed as a list to the `weight_name` parameter:

* [ip-adapter-plus_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) uses patch embeddings and a ViT-H image encoder
* [ip-adapter-plus-face_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) has the same architecture but it is conditioned with images of cropped faces

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    mindspore_dtype=ms.float16,
    image_encoder=image_encoder,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter(
  "h94/IP-Adapter",
  subfolder="sdxl_models",
  weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
)
pipeline.set_ip_adapter_scale([0.7, 0.3])
```

Load an image prompt and a folder containing images of a certain style you want to use.

```py
face_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png")
style_folder = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy"
style_images = [load_image(f"{style_folder}/img{i}.png") for i in range(10)]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1" style="width: 50%">
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image of face</figcaption>
  </div>
  <div class="flex-1" style="width: 50%">
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_style_grid.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter style images</figcaption>
  </div>
</div>

Pass the image prompt and style images as a list to the `ip_adapter_image` parameter, and run the pipeline!

```py
generator = np.random.Generator(np.random.PCG64(0))

image = pipeline(
    prompt="wonderwoman",
    ip_adapter_image=[style_images, face_image],
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=50, num_images_per_prompt=1,
    generator=generator,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/787294a4-1fe5-42dc-a20d-0872bf1f8fc1" />
</div>

### Instant generation

[Latent Consistency Models (LCM)](../using-diffusers/inference_with_lcm_lora.md) are diffusion models that can generate images in as little as 4 steps compared to other diffusion models like SDXL that typically require way more steps. This is why image generation with an LCM feels "instantaneous". IP-Adapters can be plugged into an LCM-LoRA model to instantly generate images with an image prompt.

The IP-Adapter weights need to be loaded first, then you can use [`load_lora_weights`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/text2img/#mindone.diffusers.StableDiffusionPipeline) to load the LoRA style and weight you want to apply to your image.

```py
from mindone.diffusers import DiffusionPipeline, LCMScheduler
import mindspore as ms
from mindone.diffusers.utils import load_image
import numpy as np

model_id = "sd-dreambooth-library/herge-style"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

pipeline = DiffusionPipeline.from_pretrained(model_id, mindspore_dtype=ms.float16)

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
pipeline.load_lora_weights(lcm_lora_id)
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
```

Try using with a lower IP-Adapter scale to condition image generation more on the [herge_style](https://huggingface.co/sd-dreambooth-library/herge-style) checkpoint, and remember to use the special token `herge_style` in your prompt to trigger and apply the style.

```py
pipeline.set_ip_adapter_scale(0.4)

prompt = "herge_style woman in armor, best quality, high quality"
generator = np.random.Generator(np.random.PCG64(0))

ip_adapter_image = load_image("https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
image = pipeline(
    prompt=prompt,
    ip_adapter_image=ip_adapter_image,
    num_inference_steps=4,
    guidance_scale=1,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/9b1f5b61-d431-4b69-b034-022d350e9fc8" />
</div>

### Structural control

To control image generation to an even greater degree, you can combine IP-Adapter with a model like [ControlNet](../using-diffusers/controlnet.md). A ControlNet is also an adapter that can be inserted into a diffusion model to allow for conditioning on an additional control image. The control image can be depth maps, edge maps, pose estimations, and more.

Load a [`ControlNetModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/controlnet/#mindone.diffusers.ControlNetModel) checkpoint conditioned on depth maps, insert it into a diffusion model, and load the IP-Adapter.

```py
from mindone.diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import mindspore as ms
from mindone.diffusers.utils import load_image
import numpy as np

controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, mindspore_dtype=ms.float16)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, mindspore_dtype=ms.float16)
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
```

Now load the IP-Adapter image and depth map.

```py
ip_adapter_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png")
depth_map = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/depth.png")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1" style="width: 50%">
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
  </div>
  <div class="flex-1" style="width: 50%">
    <img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/depth.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">depth map</figcaption>
  </div>
</div>

Pass the depth map and IP-Adapter image to the pipeline to generate an image.

```py
generator = np.random.Generator(np.random.PCG64(33))
image = pipeline(
    prompt="best quality, high quality",
    image=depth_map,
    ip_adapter_image=ip_adapter_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=50,
    generator=generator,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/2639c82f-2c52-4c17-8e29-c37fd8874081" />
</div>

### Style & layout control

[InstantStyle](https://arxiv.org/abs/2404.02733) is a plug-and-play method on top of IP-Adapter, which disentangles style and layout from image prompt to control image generation. This way, you can generate images following only the style or layout from image prompt, with significantly improved diversity. This is achieved by only activating IP-Adapters to specific parts of the model.

By default IP-Adapters are inserted to all layers of the model. Use the [`set_ip_adapter_scale`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/ip_adapter/#mindone.diffusers.loaders.ip_adapter.IPAdapterMixin.set_ip_adapter_scale) method with a dictionary to assign scales to IP-Adapter at different layers.

```py
from mindone.diffusers import StableDiffusionXLPipeline
from mindone.diffusers.utils import load_image
import mindspore as ms
import numpy as np

pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16)
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")

scale = {
    "down": {"block_2": [0.0, 1.0]},
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale)
```

This will activate IP-Adapter at the second layer in the model's down-part block 2 and up-part block 0. The former is the layer where IP-Adapter injects layout information and the latter injects style. Inserting IP-Adapter to these two layers you can generate images following both the style and layout from image prompt, but with contents more aligned to text prompt.

```py
style_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg")

generator = np.random.Generator(np.random.PCG64(26))
image = pipeline(
    prompt="a cat, masterpiece, best quality, high quality",
    ip_adapter_image=style_image,
    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    guidance_scale=5,
    num_inference_steps=30,
    generator=generator,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1" style="width: 50%">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter image</figcaption>
  </div>
  <div class="flex-1" style="width: 50%">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/9eb6fd5b-46f8-4be2-b23f-541a0466abb6"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

In contrast, inserting IP-Adapter to all layers will often generate images that overly focus on image prompt and diminish diversity.

Activate IP-Adapter only in the style layer and then call the pipeline again.

```py
scale = {
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale)

generator = np.random.Generator(np.random.PCG64(26))
image = pipeline(
    prompt="a cat, masterpiece, best quality, high quality",
    ip_adapter_image=style_image,
    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    guidance_scale=5,
    num_inference_steps=30,
    generator=generator,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/b4a2ca95-5a8a-404d-9307-5e3f032c383c"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter only in style layer</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/85732cd8-5f0b-4c17-a2d3-68ae7fd5b322"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">IP-Adapter in all layers</figcaption>
  </div>
</div>

Note that you don't have to specify all layers in the dictionary. Those not included in the dictionary will be set to scale 0 which means disable IP-Adapter by default.
