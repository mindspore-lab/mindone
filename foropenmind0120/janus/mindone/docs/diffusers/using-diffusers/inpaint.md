<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Inpainting

Inpainting replaces or edits specific areas of an image. This makes it a useful tool for image restoration like removing defects and artifacts, or even replacing an image area with something entirely new. Inpainting relies on a mask to determine which regions of an image to fill in; the area to inpaint is represented by white pixels and the area to keep is represented by black pixels. The white pixels are filled in by the prompt.

With 🤗 Diffusers, here is how you can do inpainting:

1. Load an inpainting checkpoint with the [`KandinskyV22InpaintCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22InpaintPipeline) class:

```py
import mindspore as ms
from mindone.diffusers import KandinskyV22InpaintCombinedPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = KandinskyV22InpaintCombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", mindspore_dtype=ms.float16
)
```

2. Load the base and mask images:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")
```

3. Create a prompt to inpaint the image with and pass it to the pipeline with the base and mask images:

```py
prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">base image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">mask image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/5d27eca4-26c5-4182-95fb-8c6c33de09b8"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

## Popular models

[Stable Diffusion Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting), [Stable Diffusion XL (SDXL) Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), and [Kandinsky 2.2 Inpainting](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint) are among the most popular models for inpainting. SDXL typically produces higher resolution images than Stable Diffusion v1.5, and Kandinsky 2.2 is also capable of generating high-quality images.

### Stable Diffusion Inpainting

Stable Diffusion Inpainting is a latent diffusion model finetuned on 512x512 images on inpainting. It is a good starting point because it is relatively fast and generates good quality images. To use this model for inpainting, you'll need to pass a prompt, base and mask image to the pipeline:

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import numpy as np

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", mindspore_dtype=ms.float16, variant="fp16"
)

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = np.random.Generator(np.random.PCG64(92))
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

### Stable Diffusion XL (SDXL) Inpainting

SDXL is a larger and more powerful version of Stable Diffusion v1.5. This model can follow a two-stage model process (though each model can also be used alone); the base model generates an image, and a refiner model takes that image and further enhances its details and quality. Take a look at the [SDXL](sdxl.md) guide for a more comprehensive guide on how to use SDXL and configure it's parameters.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import numpy as np

pipeline = DiffusionPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", mindspore_dtype=ms.float16, variant="fp16"
)

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = np.random.Generator(np.random.PCG64(92))
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

### Kandinsky 2.2 Inpainting

The Kandinsky model family is similar to SDXL because it uses two models as well; the image prior model creates image embeddings, and the diffusion model generates images from them. You can load the image prior and diffusion model separately, but the easiest way to use Kandinsky 2.2 is to load it into the [`KandinskyV22InpaintCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22InpaintCombinedPipeline) class.

```py
import mindspore as ms
from mindone.diffusers import KandinskyV22InpaintCombinedPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import numpy as np

pipeline = KandinskyV22InpaintCombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", mindspore_dtype=ms.float16
)

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

generator = np.random.Generator(np.random.PCG64(92))
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1" style="width: 25%">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">base image</figcaption>
  </div>
  <div class="flex-1" style="width: 25%">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/7732713a-a22e-4500-b485-bf90691befd9"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion Inpainting</figcaption>
  </div>
  <div class="flex-1" style="width: 25%">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/9d73ff74-6958-4bab-84eb-32ff65a4cff2"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion XL Inpainting</figcaption>
  </div>
  <div class="flex-1" style="width: 25%">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/b7144fa5-2400-4522-ad16-2dc522c1ce32"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Kandinsky 2.2 Inpainting</figcaption>
  </div>
</div>

## Non-inpaint specific checkpoints

So far, this guide has used inpaint specific checkpoints such as [stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting). But you can also use regular checkpoints like [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Let's compare the results of the two checkpoints.

The image on the left is generated from a regular checkpoint, and the image on the right is from an inpaint checkpoint. You'll immediately notice the image on the left is not as clean, and you can still see the outline of the area the model is supposed to inpaint. The image on the right is much cleaner and the inpainted area appears more natural.

=== "stable-diffusion-v1-5/stable-diffusion-v1-5"

    ```py
    import mindspore as ms
    from mindone.diffusers import StableDiffusionInpaintPipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import numpy as np

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16"
    )

    # load base and mask image
    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
    mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

    generator = np.random.Generator(np.random.PCG64(92))
    prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator)[0][0]
    make_image_grid([init_image, image], rows=1, cols=2)
    ```

=== "stable-diffusion-v1-5/stable-diffusion-inpainting"

    ```py
    import mindspore as ms
    from mindone.diffusers import DiffusionPipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import numpy as np

    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", mindspore_dtype=ms.float16, variant="fp16"
    )

    # load base and mask image
    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
    mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

    generator = np.random.Generator(np.random.PCG64(92))
    prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator)[0][0]
    make_image_grid([init_image, image], rows=1, cols=2)
    ```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/cde9c2bd-0a51-4d3b-9c55-429f11bf056a"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">stable-diffusion-v1-5/stable-diffusion-v1-5</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/182eac99-f46c-4fb1-acb0-1ead2331451f"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">stable-diffusion-v1-5/stable-diffusion-inpainting</figcaption>
  </div>
</div>

However, for more basic tasks like erasing an object from an image (like the rocks in the road for example), a regular checkpoint yields pretty good results. There isn't as noticeable of difference between the regular and inpaint checkpoint.

=== "stable-diffusion-v1-5/stable-diffusion-v1-5"

    ```py
    import mindspore as ms
    from mindone.diffusers import StableDiffusionInpaintPipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import numpy as np

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16"
    )

    # load base and mask image
    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
    mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/road-mask.png")

    image = pipeline(prompt="road", image=init_image, mask_image=mask_image)[0][0]
    make_image_grid([init_image, image], rows=1, cols=2)
    ```

=== "stable-diffusion-v1-5/stable-diffusion-inpainting"

    ```py
    import mindspore as ms
    from mindone.diffusers import DiffusionPipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import numpy as np

    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", mindspore_dtype=ms.float16, variant="fp16"
    )

    # load base and mask image
    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
    mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/road-mask.png")

    image = pipeline(prompt="road", image=init_image, mask_image=mask_image)[0][0]
    make_image_grid([init_image, image], rows=1, cols=2)
    ```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/51be158f-d5fb-47a8-9735-ae7fbad18b52"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">stable-diffusion-v1-5/stable-diffusion-v1-5</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/cf15155b-936f-40e0-bbcc-451b5cd9a9ef"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">stable-diffusion-v1-5/stable-diffusion-inpainting</figcaption>
  </div>
</div>

The trade-off of using a non-inpaint specific checkpoint is the overall image quality may be lower, but it generally tends to preserve the mask area (that is why you can see the mask outline). The inpaint specific checkpoints are intentionally trained to generate higher quality inpainted images, and that includes creating a more natural transition between the masked and unmasked areas. As a result, these checkpoints are more likely to change your unmasked area.

If preserving the unmasked area is important for your task, you can use the [`VaeImageProcessor.apply_overlay`] method to force the unmasked area of an image to remain the same at the expense of some more unnatural transitions between the masked and unmasked areas.

```py
import PIL
import numpy as np
import mindspore as ms

from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    mindspore_dtype=ms.float16,
)

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
repainted_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image)[0][0]
repainted_image.save("repainted_image.png")

unmasked_unchanged_image = pipeline.image_processor.apply_overlay(mask_image, init_image, repainted_image)
unmasked_unchanged_image.save("force_unmasked_unchanged.png")
make_image_grid([init_image, mask_image, repainted_image, unmasked_unchanged_image], rows=2, cols=2)
```

## Configure pipeline parameters

Image features - like quality and "creativity" - are dependent on pipeline parameters. Knowing what these parameters do is important for getting the results you want. Let's take a look at the most important parameters and see how changing them affects the output.

### Strength

`strength` is a measure of how much noise is added to the base image, which influences how similar the output is to the base image.

* 📈 a high `strength` value means more noise is added to an image and the denoising process takes longer, but you'll get higher quality images that are more different from the base image
* 📉 a low `strength` value means less noise is added to an image and the denoising process is faster, but the image quality may not be as great and the generated image resembles the base image more

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", mindspore_dtype=ms.float16, variant="fp16"
)

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.6)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/81dbee6c-9753-4a2b-b0f0-d16ef99ff3d2"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.6</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/a5141339-1110-4bb2-9377-de64b77105c1"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.8</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/a74ad41d-ba83-4496-bb01-d619c2721b9a"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 1.0</figcaption>
  </div>
</div>

### Guidance scale

`guidance_scale` affects how aligned the text prompt and generated image are.

* 📈 a high `guidance_scale` value means the prompt and generated image are closely aligned, so the output is a stricter interpretation of the prompt
* 📉 a low `guidance_scale` value means the prompt and generated image are more loosely aligned, so the output may be more varied from the prompt

You can use `strength` and `guidance_scale` together for more control over how expressive the model is. For example, a combination high `strength` and `guidance_scale` values gives the model the most creative freedom.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", mindspore_dtype=ms.float16, variant="fp16"
)

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, guidance_scale=2.5)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/816531e9-9bcc-46ac-9f23-3a8b0e4d221a"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 2.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/d931fec5-8a99-4507-8cff-336f56d63a0e"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 7.5</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c658249f-6bb3-479e-81d4-c0e2daec9c10"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 12.5</figcaption>
  </div>
</div>

### Negative prompt

A negative prompt assumes the opposite role of a prompt; it guides the model away from generating certain things in an image. This is useful for quickly improving image quality and preventing the model from generating things you don't want.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", mindspore_dtype=ms.float16, variant="fp16"
)

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
negative_prompt = "bad architecture, unstable, poor details, blurry"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask_image)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <figure>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/911fe268-0190-493c-a0bb-65e3a6568fa2" />
    <figcaption class="text-center">negative_prompt = "bad architecture, unstable, poor details, blurry"</figcaption>
  </figure>
</div>

### Padding mask crop

A method for increasing the inpainting image quality is to use the [`padding_mask_crop`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/inpaint/#mindone.diffusers.StableDiffusionInpaintPipeline) parameter. When enabled, this option crops the masked area with some user-specified padding and it'll also crop the same area from the original image. Both the image and mask are upscaled to a higher resolution for inpainting, and then overlaid on the original image. This is a quick and easy way to improve image quality without using a separate pipeline like [`StableDiffusionUpscalePipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/upscale/#mindone.diffusers.StableDiffusionUpscalePipeline).

Add the `padding_mask_crop` parameter to the pipeline call and set it to the desired padding value.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionInpaintPipeline
from mindone.diffusers.utils import load_image
import numpy as np
from PIL import Image

generator = np.random.Generator(np.random.PCG64(0))
pipeline = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16)

base = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png")
mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore_mask.png")

image = pipeline("boat", image=base, mask_image=mask, strength=0.75, generator=generator, padding_mask_crop=32)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/bd3bd105-8487-4135-9f80-6d67e10716dd"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">default inpaint image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/6a2ae70a-7364-48f9-8e69-f3637a736386"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">inpaint image with `padding_mask_crop` enabled</figcaption>
  </div>
</div>

## Chained inpainting pipelines

[`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) can be chained with other 🤗 Diffusers pipelines to edit their outputs. This is often useful for improving the output quality from your other diffusion pipelines, and if you're using multiple pipelines, it can be more memory-efficient to chain them together to keep the outputs in latent space and reuse the same pipeline components.

### Text-to-image-to-inpaint

Chaining a text-to-image and inpainting pipeline allows you to inpaint the generated image, and you don't have to provide a base image to begin with. This makes it convenient to edit your favorite text-to-image outputs without having to generate an entirely new image.

Start with the text-to-image pipeline to create a castle:

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline, KandinskyV22InpaintCombinedPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

text2image = pipeline("concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k")[0][0]
```

Load the mask image of the output from above:

```py
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_text-chain-mask.png")
```

And let's inpaint the masked area with a waterfall:

```py
pipeline = KandinskyV22InpaintCombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", mindspore_dtype=ms.float16
)

prompt = "digital painting of a fantasy waterfall, cloudy"
image = pipeline(prompt=prompt, image=text2image, mask_image=mask_image)[0][0]
make_image_grid([text2image, mask_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/d3ac2422-d674-4991-9bc4-ea719b776144"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">text-to-image</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/a4c4b149-2929-4b94-a724-086efce4e5f3"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">inpaint</figcaption>
  </div>
</div>

### Inpaint-to-image-to-image

You can also chain an inpainting pipeline before another pipeline like image-to-image or an upscaler to improve the quality.

Begin by inpainting an image:

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline, KandinskyV22InpaintCombinedPipeline, StableDiffusionInpaintPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", mindspore_dtype=ms.float16, variant="fp16"
)

# load base and mask image
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
image_inpainting = pipeline(prompt=prompt, image=init_image, mask_image=mask_image)[0][0]

# resize image to 1024x1024 for SDXL
image_inpainting = image_inpainting.resize((1024, 1024))
```

Now let's pass the image to another inpainting pipeline with SDXL's refiner model to enhance the image details and quality:

```py
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", mindspore_dtype=ms.float16, variant="fp16"
)

image = pipeline(prompt=prompt, image=image_inpainting, mask_image=mask_image, output_type="latent")[0][0]
```

!!! tip

    It is important to specify `output_type="latent"` in the pipeline to keep all the outputs in latent space to avoid an unnecessary decode-encode step. This only works if the chained pipelines are using the same VAE. For example, in the [Text-to-image-to-inpaint](#text-to-image-to-inpaint.md) section, Kandinsky 2.2 uses a different VAE class than the Stable Diffusion model so it won't work. But if you use Stable Diffusion v1.5 for both pipelines, then you can keep everything in latent space because they both use [`AutoencoderKL`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/autoencoderkl/#mindone.diffusers.AutoencoderKL).

Finally, you can pass this image to an image-to-image pipeline to put the finishing touches on it.

```py
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", mindspore_dtype=ms.float16, variant="fp16"
)

image = pipeline(prompt=prompt, image=image)[0][0]
make_image_grid([init_image, mask_image, image_inpainting, image], rows=2, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1" style="width: 33%">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div class="flex-1" style="width: 33%">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/cf900697-2358-4559-a3b2-f0c173415d6b"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">inpaint</figcaption>
  </div>
  <div class="flex-1" style="width: 33%">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/8226d747-d122-4199-92bc-69b85263bc10"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">image-to-image</figcaption>
  </div>
</div>

Image-to-image and inpainting are actually very similar tasks. Image-to-image generates a new image that resembles the existing provided image. Inpainting does the same thing, but it only transforms the image area defined by the mask and the rest of the image is unchanged. You can think of inpainting as a more precise tool for making specific changes and image-to-image has a broader scope for making more sweeping changes.
