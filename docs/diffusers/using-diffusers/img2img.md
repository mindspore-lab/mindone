<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Image-to-image

Image-to-image is similar to [text-to-image](conditional_image_generation.md), but in addition to a prompt, you can also pass an initial image as a starting point for the diffusion process. The initial image is encoded to latent space and noise is added to it. Then the latent diffusion model takes a prompt and the noisy latent image, predicts the added noise, and removes the predicted noise from the initial latent image to get the new latent image. Lastly, a decoder decodes the new latent image back into an image.

With 🤗 Diffusers, this is as easy as 1-2-3:

1. Load a checkpoint into the [`KandinskyV22Img2ImgCombinedPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22Img2ImgCombinedPipeline) class:

```py
import mindspore as ms
from mindone.diffusers import KandinskyV22Img2ImgCombinedPipeline
from mindone.diffusers.utils import load_image, make_image_grid

pipeline = KandinskyV22Img2ImgCombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16, use_safetensors=True
)
```

2. Load an image to pass to the pipeline:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
```

3. Pass a prompt and image to the pipeline to generate an image:

```py
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/4baf9655-0e26-4e90-a248-c7369e85e072"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

## Popular models

The most popular image-to-image models are [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), and [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder). The results from the Stable Diffusion and Kandinsky models vary due to their architecture differences and training process; you can generally expect SDXL to produce higher quality images than Stable Diffusion v1.5. Let's take a quick look at how to use each of these models and compare their results.

### Stable Diffusion v1.5

Stable Diffusion v1.5 is a latent diffusion model initialized from an earlier checkpoint, and further finetuned for 595K steps on 512x512 images. To use this pipeline for image-to-image, you'll need to prepare an initial image to pass to the pipeline. Then you can pass a prompt and the image to the pipeline to generate a new image:

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionImg2ImgPipeline
from mindone.diffusers.utils import make_image_grid, load_image

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/b0323682-c4e1-475f-bea2-6845c5984768"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

### Stable Diffusion XL (SDXL)

SDXL is a more powerful version of the Stable Diffusion model. It uses a larger base model, and an additional refiner model to increase the quality of the base model's output. Read the [SDXL](sdxl.md) guide for a more detailed walkthrough of how to use this model, and other techniques it uses to produce high quality images.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import make_image_grid, load_image

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.5)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/a1d19267-43d4-4093-b56a-ac8d4640cce9"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

### Kandinsky 2.2

The Kandinsky model is different from the Stable Diffusion models because it uses an image prior model to create image embeddings. The embeddings help create a better alignment between text and images, allowing the latent diffusion model to generate better images.

The simplest way to use Kandinsky 2.2 is:

```py
import mindspore as ms
from mindone.diffusers import KandinskyV22Img2ImgCombinedPipeline
from mindone.diffusers.utils import make_image_grid, load_image

pipeline = KandinskyV22Img2ImgCombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16, use_safetensors=True
)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">initial image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c3b993ab-86e3-40ad-9138-1a422eb3a9a0"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">generated image</figcaption>
  </div>
</div>

## Configure pipeline parameters

There are several important parameters you can configure in the pipeline that'll affect the image generation process and image quality. Let's take a closer look at what these parameters do and how changing them affects the output.

### Strength

`strength` is one of the most important parameters to consider and it'll have a huge impact on your generated image. It determines how much the generated image resembles the initial image. In other words:

- 📈 a higher `strength` value gives the model more "creativity" to generate an image that's different from the initial image; a `strength` value of 1.0 means the initial image is more or less ignored
- 📉 a lower `strength` value means the generated image is more similar to the initial image

The `strength` and `num_inference_steps` parameters are related because `strength` determines the number of noise steps to add. For example, if the `num_inference_steps` is 50 and `strength` is 0.8, then this means adding 40 (50 * 0.8) steps of noise to the initial image and then denoising for 40 steps to get the newly generated image.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionImg2ImgPipeline
from mindone.diffusers.utils import make_image_grid, load_image

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, strength=0.8)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/cf83c38e-8159-4ba7-8635-8b238906a281"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.4</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/e5d8ff4d-67a6-4291-bff5-1d0e25f362ea"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 0.6</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/54271a14-52fe-4ab1-9fa2-6809b9dd8c05"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">strength = 1.0</figcaption>
  </div>
</div>

### Guidance scale

The `guidance_scale` parameter is used to control how closely aligned the generated image and text prompt are. A higher `guidance_scale` value means your generated image is more aligned with the prompt, while a lower `guidance_scale` value means your generated image has more space to deviate from the prompt.

You can combine `guidance_scale` with `strength` for even more precise control over how expressive the model is. For example, combine a high `strength + guidance_scale` for maximum creativity or use a combination of low `strength` and low `guidance_scale` to generate an image that resembles the initial image but is not as strictly bound to the prompt.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionImg2ImgPipeline
from mindone.diffusers.utils import make_image_grid, load_image

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, guidance_scale=8.0)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/2ea09019-d28e-41e0-9e62-09bfa56d8830"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 0.1</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/7b934381-d086-4b64-9f8b-451ff584dbbb"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 5.0</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/d98cd1e0-95f4-4f85-9402-e159a772b4fd"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">guidance_scale = 10.0</figcaption>
  </div>
</div>

### Negative prompt

A negative prompt conditions the model to *not* include things in an image, and it can be used to improve image quality or modify an image. For example, you can improve image quality by including negative prompts like "poor details" or "blurry" to encourage the model to generate a higher quality image. Or you can modify an image by specifying things to exclude from an image.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import make_image_grid, load_image

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

# pass prompt and image to pipeline
image = pipeline(prompt, negative_prompt=negative_prompt, image=init_image)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/be8da336-dda9-45a1-87e7-d1c16d8e9a3d"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"</figcaption>
  </div>
  <div class="flex-1">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/441cece4-d0a5-4ea0-ba7e-ed4cd389801b"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">negative_prompt = "jungle"</figcaption>
  </div>
</div>

## Chained image-to-image pipelines

There are some other interesting ways you can use an image-to-image pipeline aside from just generating an image (although that is pretty cool too). You can take it a step further and chain it with other pipelines.

### Text-to-image-to-image

Chaining a text-to-image and image-to-image pipeline allows you to generate an image from text and use the generated image as the initial image for the image-to-image pipeline. This is useful if you want to generate an image entirely from scratch. For example, let's chain a Stable Diffusion and a Kandinsky model.

Start by generating an image with the text-to-image pipeline:

```py
from mindone.diffusers import DiffusionPipeline, KandinskyV22Img2ImgCombinedPipeline
import mindspore as ms
from mindone.diffusers.utils import make_image_grid

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

text2image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k")[0][0]
text2image
```

Now you can pass this generated image to the image-to-image pipeline:

```py
pipeline = KandinskyV22Img2ImgCombinedPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16, use_safetensors=True
)

image2image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=text2image)[0][0]
make_image_grid([text2image, image2image], rows=1, cols=2)
```

### Image-to-image-to-image

You can also chain multiple image-to-image pipelines together to create more interesting images. This can be useful for iteratively performing style transfer on an image, generating short GIFs, restoring color to an image, or restoring missing areas of an image.

Start by generating an image:

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils import make_image_grid, load_image

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, variant="fp16", use_safetensors=True
)

# prepare image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# pass prompt and image to pipeline
image = pipeline(prompt, image=init_image, output_type="latent")[0][0]
```

!!! tip

    It is important to specify `output_type="latent"` in the pipeline to keep all the outputs in latent space to avoid an unnecessary decode-encode step. This only works if the chained pipelines are using the same VAE.

Pass the latent output from this pipeline to the next pipeline to generate an image in a [comic book art style](https://huggingface.co/ogkalu/Comic-Diffusion):

```py
pipeline = DiffusionPipeline.from_pretrained(
    "ogkalu/Comic-Diffusion", mindspore_dtype=ms.float16, use_safetensors=True
)

# need to include the token "charliebo artstyle" in the prompt to use this checkpoint
image = pipeline("Astronaut in a jungle, charliebo artstyle", image=image, output_type="latent")[0][0]
```

Repeat one more time to generate the final image in a [pixel art style](https://huggingface.co/kohbanye/pixel-art-style):

```py
pipeline = DiffusionPipeline.from_pretrained(
    "kohbanye/pixel-art-style", mindspore_dtype=ms.float16, use_safetensors=True
)

# need to include the token "pixelartstyle" in the prompt to use this checkpoint
image = pipeline("Astronaut in a jungle, pixelartstyle", image=image)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```
