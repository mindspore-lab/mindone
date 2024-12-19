<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky

The Kandinsky models are a series of multilingual text-to-image generation models. The Kandinsky 2.0 model uses two multilingual text encoders and concatenates those results for the UNet.

[Kandinsky 2.1](../api/pipelines/kandinsky.md) changes the architecture to include an image prior model ([`CLIP`](https://huggingface.co/docs/transformers/model_doc/clip)) to generate a mapping between text and image embeddings. The mapping provides better text-image alignment and it is used with the text embeddings during training, leading to higher quality results. Finally, Kandinsky 2.1 uses a [Modulating Quantized Vectors (MoVQ)](https://huggingface.co/papers/2209.09002) decoder - which adds a spatial conditional normalization layer to increase photorealism - to decode the latents into images.

[Kandinsky 2.2](../api/pipelines/kandinsky_v22.md) improves on the previous model by replacing the image encoder of the image prior model with a larger CLIP-ViT-G model to improve quality. The image prior model was also retrained on images with different resolutions and aspect ratios to generate higher-resolution images and different image sizes.

[Kandinsky 3](../api/pipelines/kandinsky3.md) simplifies the architecture and shifts away from the two-stage generation process involving the prior model and diffusion model. Instead, Kandinsky 3 uses [Flan-UL2](https://huggingface.co/google/flan-ul2) to encode text, a UNet with [BigGan-deep](https://hf.co/papers/1809.11096) blocks, and [Sber-MoVQGAN](https://github.com/ai-forever/MoVQGAN) to decode the latents into images. Text understanding and generated image quality are primarily achieved by using a larger text encoder and UNet.

This guide will show you how to use the Kandinsky models for text-to-image, image-to-image, inpainting, interpolation, and more.

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries
#!pip install mindone transformers
```

!!! warning

    Kandinsky 2.1 and 2.2 usage is very similar! The only difference is Kandinsky 2.2 doesn't accept `prompt` as an input when decoding the latents. Instead, Kandinsky 2.2 only accepts `image_embeds` during decoding.

    Kandinsky 3 has a more concise architecture and it doesn't require a prior model. This means it's usage is identical to other diffusion models like [Stable Diffusion XL](sdxl.md).

    Additionally, Kandinsky 3 has precision issues now. Please refer to the [Limitation](../limitations.md) for further details.

## Text-to-image

To use the Kandinsky models for any task, you always start by setting up the prior pipeline to encode the prompt and generate the image embeddings. The prior pipeline also generates `negative_image_embeds` that correspond to the negative prompt `""`. For better results, you can pass an actual `negative_prompt` to the prior pipeline, but this'll increase the effective batch size of the prior pipeline by 2x.

=== "Kandinsky 2.1"

    ```py
    from mindone.diffusers import KandinskyPriorPipeline, KandinskyPipeline
    import mindspore as ms

    prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms.float16)
    pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms.float16)

    prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
    negative_prompt = "low quality, bad quality" # optional to include a negative prompt, but results are usually better
    image_embeds, negative_image_embeds = prior_pipeline(prompt, negative_prompt, guidance_scale=1.0)
    ```

    Now pass all the prompts and embeddings to the [`KandinskyPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky/#mindone.diffusers.KandinskyPipeline) to generate an image:

    ```py
    image = pipeline(prompt, image_embeds=image_embeds, negative_prompt=negative_prompt, negative_image_embeds=negative_image_embeds, height=768, width=768)[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/ec761cf3-8781-4ae7-8b06-c6940856f45f"/>
    </div>

=== "Kandinsky 2.2"

    ```py
    from mindone.diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
    import mindspore as ms

    prior_pipeline = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", mindspore=ms.float16)
    pipeline = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16)

    prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
    negative_prompt = "low quality, bad quality" # optional to include a negative prompt, but results are usually better
    image_embeds, negative_image_embeds = prior_pipeline(prompt, guidance_scale=1.0)
    ```

    Pass the `image_embeds` and `negative_image_embeds` to the [`KandinskyV22Pipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22Pipeline) to generate an image:

    ```py
    image = pipeline(image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=768, width=768)[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/343d369f-55e0-43b6-a48c-675ead75c524"/>
    </div>

=== "Kandinsky 3"

    Kandinsky 3 doesn't require a prior model so you can directly load the [`Kandinsky3Pipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky3/#mindone.diffusers.Kandinsky3Pipeline) and pass a prompt to generate an image:

    ```py
    from mindone.diffusers import Kandinsky3Pipeline
    import mindspore as ms

    pipeline = Kandinsky3Pipeline.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", mindspore_dtype=ms.float16)

    prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
    image = pipeline(prompt)[0][0]
    image
    ```

🤗 Diffusers also provides an end-to-end API with the [`KandinskyCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky/#mindone.diffusers.KandinskyCombinedPipeline) and [`KandinskyV22CombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22CombinedPipeline), meaning you don't have to separately load the prior and text-to-image pipeline. The combined pipeline automatically loads both the prior model and the decoder. You can still set different values for the prior pipeline with the `prior_guidance_scale` and `prior_num_inference_steps` parameters if you want.

=== "Kandinsky 2.1"

    ```py
    from mindone.diffusers import KandinskyCombinedPipeline
    import mindspore as ms

    pipeline = KandinskyCombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms.float16)

    prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
    negative_prompt = "low quality, bad quality"

    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, prior_guidance_scale=1.0, guidance_scale=4.0, height=768, width=768)[0][0]
    image
    ```

=== "Kandinsky 2.2"

    ```py
    from mindone.diffusers import KandinskyV22CombinedPipeline
    import mindspore as ms

    pipeline = KandinskyV22CombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16)

    prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
    negative_prompt = "low quality, bad quality"

    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, prior_guidance_scale=1.0, guidance_scale=4.0, height=768, width=768)[0][0]
    image
    ```

## Image-to-image

For image-to-image, pass the initial image and text prompt to condition the image to the pipeline. Start by loading the prior pipeline:

=== "Kandinsky 2.1"

    ```py
    import mindspore as ms
    from mindone.diffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline

    prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms.float16, use_safetensors=True)
    pipeline = KandinskyImg2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms.float16, use_safetensors=True)
    ```

=== "Kandinsky 2.2"

    ```py
    import mindspore as ms
    from mindone.diffusers import KandinskyV22Img2ImgPipeline, KandinskyPriorPipeline

    prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16, use_safetensors=True)
    pipeline = KandinskyV22Img2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16, use_safetensors=True)
    ```

=== "Kandinsky 3"

    Kandinsky 3 doesn't require a prior model so you can directly load the image-to-image pipeline:

    ```py
    from mindone.diffusers import Kandinsky3Img2ImgPipeline
    from mindone.diffusers.utils import load_image
    import mindspore as ms

    pipeline = Kandinsky3Img2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", mindspore_dtype=ms.float16)
    ```

Download an image to condition on:

```py
from mindone.diffusers.utils import load_image

# download image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
original_image = load_image(url)
original_image = original_image.resize((768, 512))
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"/>
</div>

Generate the `image_embeds` and `negative_image_embeds` with the prior pipeline:

```py
prompt = "A fantasy landscape, Cinematic lighting"
negative_prompt = "low quality, bad quality"

image_embeds, negative_image_embeds = prior_pipeline(prompt, negative_prompt)
```

Now pass the original image, and all the prompts and embeddings to the pipeline to generate an image:

=== "Kandinsky 2.1"

    ```py
    from mindone.diffusers.utils import make_image_grid

    image = pipeline(prompt, negative_prompt=negative_prompt, image=original_image, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=768, width=768, strength=0.3)[0][0]
    make_image_grid([original_image.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/0ab742d0-1dcd-449c-883c-215f80fdc9e1"/>
    </div>

=== "Kandinsky 2.2"

    ```py
    from mindone.diffusers.utils import make_image_grid

    image = pipeline(image=original_image, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=768, width=768, strength=0.3)[0][0]
    make_image_grid([original_image.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/df06d63d-3cd6-44bf-aa0b-eb81d4c5cef1"/>
    </div>

=== "Kandinsky 3"

    ```py
    image = pipeline(prompt, negative_prompt=negative_prompt, image=original_image, strength=0.75, num_inference_steps=25)[0][0]
    image
    ```

🤗 Diffusers also provides an end-to-end API with the [`KandinskyImg2ImgCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky/#mindone.diffusers.KandinskyImg2ImgCombinedPipeline) and [`KandinskyV22Img2ImgCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22Img2ImgCombinedPipeline), meaning you don't have to separately load the prior and image-to-image pipeline. The combined pipeline automatically loads both the prior model and the decoder. You can still set different values for the prior pipeline with the `prior_guidance_scale` and `prior_num_inference_steps` parameters if you want.

=== "Kandinsky 2.1"

    ```py
    from mindone.diffusers import KandinskyImg2ImgCombinedPipeline
    from mindone.diffusers.utils import make_image_grid, load_image
    import mindspore as ms

    pipeline = KandinskyImg2ImgCombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms.float16, use_safetensors=True)

    prompt = "A fantasy landscape, Cinematic lighting"
    negative_prompt = "low quality, bad quality"

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    original_image = load_image(url)

    original_image.thumbnail((768, 768))

    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=original_image, strength=0.3)[0][0]
    make_image_grid([original_image.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
    ```

=== "Kandinsky 2.2"

    ```py
    from mindone.diffusers import KandinskyV22Img2ImgCombinedPipeline
    from mindone.diffusers.utils import make_image_grid, load_image
    import mindspore as ms

    pipeline = KandinskyV22Img2ImgCombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16)

    prompt = "A fantasy landscape, Cinematic lighting"
    negative_prompt = "low quality, bad quality"

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    original_image = load_image(url)

    original_image.thumbnail((768, 768))

    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=original_image, strength=0.3)[0][0]
    make_image_grid([original_image.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
    ```

## Inpainting

!!! warning

    ⚠️ The Kandinsky models use ⬜️ **white pixels** to represent the masked area now instead of black pixels. If you are using [`KandinskyInpaintPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky/#mindone.diffusers.KandinskyInpaintPipeline) in production, you need to change the mask to use white pixels:

    ```py
    # For PIL input
    import PIL.ImageOps
    mask = PIL.ImageOps.invert(mask)

    # For MindSpore and NumPy input
    mask = 1 - mask
    ```

For inpainting, you'll need the original image, a mask of the area to replace in the original image, and a text prompt of what to inpaint. Load the prior pipeline:

=== "Kandinsky 2.1"

    ```py
    from mindone.diffusers import KandinskyInpaintPipeline, KandinskyPriorPipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import mindspore as ms
    import numpy as np
    from PIL import Image

    prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms.float16, use_safetensors=True)
    pipeline = KandinskyInpaintPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", mindspore_dtype=ms.float16, use_safetensors=True)
    ```

=== "Kandinsky 2.2"

    ```py
    from mindone.diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import mindspore as ms
    import numpy as np
    from PIL import Image

    prior_pipeline = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16, use_safetensors=True)
    pipeline = KandinskyV22InpaintPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", mindspore_dtype=ms.float16, use_safetensors=True)
    ```

Load an initial image and create a mask:

```py
init_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
mask = np.zeros((768, 768), dtype=np.float32)
# mask area above cat's head
mask[:250, 250:-250] = 1
```

Generate the embeddings with the prior pipeline:

```py
prompt = "a hat"
image_emb, zero_image_emb = prior_pipeline(prompt)
```

Now pass the initial image, mask, and prompt and embeddings to the pipeline to generate an image:

=== "Kandinsky 2.1"

    ```py
    output_image = pipeline(
        prompt,
        image=init_image,
        mask_image=mask,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=150
    )[0][0]
    mask = Image.fromarray((mask*255).astype('uint8'), 'L')
    make_image_grid([init_image, mask, output_image], rows=1, cols=3)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/af2022b7-d602-4352-9406-e65f2aab80ff"/>
    </div>

=== "Kandinsky 2.2"

    ```py
    output_image = pipeline(
        image=init_image,
        mask_image=mask,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=150
    )[0][0]
    mask = Image.fromarray((mask*255).astype('uint8'), 'L')
    make_image_grid([init_image, mask, output_image], rows=1, cols=3)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/a27f8001-5cb3-4d95-be31-49ee303b4089"/>
    </div>

You can also use the end-to-end [`KandinskyInpaintCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky/#mindone.diffusers.KandinskyInpaintCombinedPipeline) and [`KandinskyV22InpaintCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22InpaintPipeline) to call the prior and decoder pipelines together under the hood.

=== "Kandinsky 2.1"

    ```py
    import mindspore as ms
    import numpy as np
    from PIL import Image
    from mindone.diffusers import KandinskyInpaintCombinedPipeline
    from mindone.diffusers.utils import load_image, make_image_grid

    pipe = KandinskyInpaintCombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", mindspore_dtype=ms.float16)

    init_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
    mask = np.zeros((768, 768), dtype=np.float32)
    # mask area above cat's head
    mask[:250, 250:-250] = 1
    prompt = "a hat"

    output_image = pipe(prompt=prompt, image=init_image, mask_image=mask)[0][0]
    mask = Image.fromarray((mask*255).astype('uint8'), 'L')
    make_image_grid([init_image, mask, output_image], rows=1, cols=3)
    ```

=== "Kandinsky 2.2"

    ```py
    import mindspore as ms
    import numpy as np
    from PIL import Image
    from mindone.diffusers import KandinskyV22InpaintCombinedPipeline
    from mindone.diffusers.utils import load_image, make_image_grid

    pipe = KandinskyV22InpaintCombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", mindspore_dtype=ms.float16)

    init_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
    mask = np.zeros((768, 768), dtype=np.float32)
    # mask area above cat's head
    mask[:250, 250:-250] = 1
    prompt = "a hat"

    output_image = pipe(prompt=prompt, image=init_image, mask_image=mask)[0][0]
    mask = Image.fromarray((mask*255).astype('uint8'), 'L')
    make_image_grid([init_image, mask, output_image], rows=1, cols=3)
    ```

## Interpolation

Interpolation allows you to explore the latent space between the image and text embeddings which is a cool way to see some of the prior model's intermediate outputs. Load the prior pipeline and two images you'd like to interpolate:

=== "Kandinsky 2.1"

    ```py
    from mindone.diffusers import KandinskyPriorPipeline, KandinskyPipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import mindspore as ms

    prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms.float16, use_safetensors=True)
    img_1 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
    img_2 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg")
    make_image_grid([img_1.resize((512,512)), img_2.resize((512,512))], rows=1, cols=2)
    ```

=== "Kandinsky 2.2"

    ```py
    from mindone.diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
    from mindone.diffusers.utils import load_image, make_image_grid
    import mindspore as ms

    prior_pipeline = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16, use_safetensors=True)
    img_1 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
    img_2 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg")
    make_image_grid([img_1.resize((512,512)), img_2.resize((512,512))], rows=1, cols=2)
    ```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">a cat</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Van Gogh's Starry Night painting</figcaption>
  </div>
</div>

Specify the text or images to interpolate, and set the weights for each text or image. Experiment with the weights to see how they affect the interpolation!

```py
images_texts = ["a cat", img_1, img_2]
weights = [0.3, 0.3, 0.4]
```

Call the `interpolate` function to generate the embeddings, and then pass them to the pipeline to generate the image:

=== "Kandinsky 2.1"

    ```py
    # prompt can be left empty
    prompt = ""
    image_embeds, negative_image_embeds = prior_pipeline.interpolate(images_texts, weights)

    pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms.float16, use_safetensors=True)

    image = pipeline(prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=768, width=768)[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/056b7f15-72be-46cb-ba0e-8e55bc010685"/>
    </div>

=== "Kandinsky 2.2"

    ```py
    # prompt can be left empty
    prompt = ""
    image_embeds, negative_image_embeds = prior_pipeline.interpolate(images_texts, weights)

    pipeline = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16, use_safetensors=True)

    image = pipeline(image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=768, width=768)[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/4cad5e59-36f0-4b27-9bc9-6cb269074ebd"/>
    </div>

## ControlNet

!!! warning

    ⚠️ ControlNet is only supported for Kandinsky 2.2!

    ⚠️ MindONE currently does not support the full process for extracting the depth map, as MindONE does not yet support depth-estimation [~transformers.Pipeline] from mindone.transformers. Therefore, you need to prepare the depth map in advance to continue the process.

ControlNet enables conditioning large pretrained diffusion models with additional inputs such as a depth map or edge detection. For example, you can condition Kandinsky 2.2 with a depth map so the model understands and preserves the structure of the depth image.

Let's load an image and extract it's depth map:

```py
from mindone.diffusers.utils import load_image

img = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png"
).resize((768, 768))
img
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png"/>
</div>

Then you can process and retrieve the depth map you prepared in advance:

```py
import mindspore as ms
import numpy as np

def make_hint(depth_image):
    image = depth_image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = ms.Tensor.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint

hint = make_hint(depth_image).unsqueeze(0).half()
```

### Text-to-image

Load the prior pipeline and the [`KandinskyV22ControlnetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetPipeline):

```py
from mindone.diffusers import KandinskyV22PriorPipeline, KandinskyV22ControlnetPipeline
import mindspore as ms
import numpy as np

prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16, use_safetensors=True
)

pipeline = KandinskyV22ControlnetPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-controlnet-depth", revision="refs/pr/7", mindspore_dtype=ms.float16
)
```

Generate the image embeddings from a prompt and negative prompt:

```py
prompt = "A robot, 4k photo"
negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

generator = np.random.Generator(np.random.PCG64(43))

image_emb, zero_image_emb = prior_pipeline(
    prompt=prompt, negative_prompt=negative_prior_prompt, generator=generator
)
```

Finally, pass the image embeddings and the depth image to the [`KandinskyV22ControlnetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetPipeline) to generate an image:

```py
image = pipeline(image_embeds=image_emb, negative_image_embeds=zero_image_emb, hint=hint, num_inference_steps=50, generator=generator, height=768, width=768)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c2a92975-2596-42eb-82c9-da56f55f15a8"/>
</div>

### Image-to-image

For image-to-image with ControlNet, you'll need to use the:

- [`KandinskyV22PriorEmb2EmbPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22PriorEmb2EmbPipeline) to generate the image embeddings from a text prompt and an image
- [`KandinskyV22ControlnetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetImg2ImgPipeline) to generate an image from the initial image and the image embeddings

Process the depth map extracted from the initial image of a cat, which you prepared in advance.

```py
import mindspore as ms
import numpy as np

from mindone.diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
from mindone.diffusers.utils import load_image

img = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png"
).resize((768, 768))

def make_hint(depth_image):
    image = depth_image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = ms.Tensor.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint

hint = make_hint(depth_image).unsqueeze(0).half()
```

Load the prior pipeline and the [`KandinskyV22ControlnetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetImg2ImgPipeline):

```py
prior_pipeline = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16, use_safetensors=True
)

pipeline = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-controlnet-depth", revision="refs/pr/7", mindspore_dtype=ms.float16
)
```

Pass a text prompt and the initial image to the prior pipeline to generate the image embeddings:

```py
prompt = "A robot, 4k photo"
negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

generator = np.random.Generator(np.random.PCG64(43))

img_emb = prior_pipeline(prompt=prompt, image=img, strength=0.85, generator=generator)
negative_emb = prior_pipeline(prompt=negative_prior_prompt, image=img, strength=1, generator=generator)
```

Now you can run the [`KandinskyV22ControlnetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetImg2ImgPipeline) to generate an image from the initial image and the image embeddings:

```py
image = pipeline(image=img, strength=0.5, image_embeds=img_emb[0], negative_image_embeds=negative_emb[0], hint=hint, num_inference_steps=50, generator=generator, height=768, width=768)[0][0]
make_image_grid([img.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/b67abe65-5874-4795-bb74-3eff17e911ca"/>
</div>

## Optimizations

Kandinsky is unique because it requires a prior pipeline to generate the mappings, and a second pipeline to decode the latents into an image. Optimization efforts should be focused on the second pipeline because that is where the bulk of the computation is done. Here are some tip to improve Kandinsky during inference.

1. By default, the text-to-image pipeline uses the [`DDIMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddim/#mindone.diffusers.DDIMScheduler) but you can replace it with another scheduler like [`DDPMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddpm/#mindone.diffusers.DDPMScheduler) to see how that affects the tradeoff between inference speed and image quality:

```py
from mindone.diffusers import DDPMScheduler
from mindone.diffusers import KandinskyCombinedPipeline
import mindspore as ms

scheduler = DDPMScheduler.from_pretrained("kandinsky-community/kandinsky-2-1", subfolder="ddpm_scheduler")
pipe = KandinskyCombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", scheduler=scheduler, mindspore_dtype=ms.float16, use_safetensors=True)
```
