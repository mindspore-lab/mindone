<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Stable Diffusion XL

[Stable Diffusion XL](https://huggingface.co/papers/2307.01952) (SDXL) is a powerful text-to-image generation model that iterates on the previous Stable Diffusion models in three key ways:

1. the UNet is 3x larger and SDXL combines a second text encoder (OpenCLIP ViT-bigG/14) with the original text encoder to significantly increase the number of parameters
2. introduces size and crop-conditioning to preserve training data from being discarded and gain more control over how a generated image should be cropped
3. introduces a two-stage model process; the *base* model (can also be run as a standalone model) generates an image as an input to the *refiner* model which adds additional high-quality details

This guide will show you how to use SDXL for text-to-image, image-to-image, and inpainting.

Before you begin, make sure you have the following libraries installed:

```py
# uncomment to install the necessary libraries
#!pip install mindone transformers
```

!!! warning

    mindone.diffusers does not support watermarker to help identify generated images.

## Load model checkpoints

Model weights may be stored in separate subfolders on the Hub or locally, in which case, you should use the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) method:

```py
from mindone.diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import mindspore as ms

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)
```

You can also use the [`from_single_file`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/single_file/#mindone.diffusers.loaders.single_file.FromSingleFileMixin.from_single_file) method to load a model checkpoint stored in a single file format (`.ckpt` or `.safetensors`) from the Hub or locally:

```py
from mindone.diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import mindspore as ms

pipeline = StableDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
    mindspore_dtype=ms.float16
)

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors", mindspore_dtype=ms.float16
)
```

## Text-to-image

For text-to-image, pass a text prompt. By default, SDXL generates a 1024x1024 image for the best results. You can try setting the `height` and `width` parameters to 768x768 or 512x512, but anything below 512x512 is not likely to work.

```py
from mindone.diffusers import StableDiffusionXLPipeline
import mindspore as ms

pipeline_text2image = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/8a214aab-d479-44e0-8a86-43246cdcaeeb"/>
</div>

## Image-to-image

For image-to-image, SDXL works especially well with image sizes between 768x768 and 1024x1024. Pass an initial image, and a text prompt to condition the image with:

```py
from mindone.diffusers import StableDiffusionXLImg2ImgPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import mindspore as ms

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
init_image = load_image(url)
prompt = "a dog catching a frisbee in the jungle"
image = pipeline(prompt, image=init_image, strength=0.8, guidance_scale=10.5)[0][0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/6578a382-652e-4779-b623-9c4c9609e7d4"/>
</div>

## Inpainting

For inpainting, you'll need the original image and a mask of what you want to replace in the original image. Create a prompt to describe what you want to replace the masked area with.

```py
from mindone.diffusers import StableDiffusionXLInpaintPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import mindspore as ms

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "A deep sea diver floating"
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.85, guidance_scale=12.5)[0][0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/56dc0e21-6146-4772-8e48-05209f169ab4"/>
</div>

## Refine image quality

SDXL includes a [refiner model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) specialized in denoising low-noise stage images to generate higher-quality images from the base model. There are two ways to use the refiner:

1. use the base and refiner models together to produce a refined image
2. use the base model to produce an image, and subsequently use the refiner model to add more details to the image (this is how SDXL was originally trained)

### Base + refiner model

When you use the base and refiner model together to generate an image, this is known as an [*ensemble of expert denoisers*](https://research.nvidia.com/labs/dir/eDiff-I/). The ensemble of expert denoisers approach requires fewer overall denoising steps versus passing the base model's output to the refiner model, so it should be significantly faster to run. However, you won't be able to inspect the base model's output because it still contains a large amount of noise.

As an ensemble of expert denoisers, the base model serves as the expert during the high-noise diffusion stage and the refiner model serves as the expert during the low-noise diffusion stage. Load the base and refiner model:

```py
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    mindspore_dtype=ms.float16,
    use_safetensors=True,
)
```

To use this approach, you need to define the number of timesteps for each model to run through their respective stages. For the base model, this is controlled by the [`denoising_end`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLPipeline) parameter and for the refiner model, it is controlled by the [`denoising_start`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLImg2ImgPipeline) parameter.

!!! tip

    The `denoising_end` and `denoising_start` parameters should be a float between 0 and 1. These parameters are represented as a proportion of discrete timesteps as defined by the scheduler. If you're also using the `strength` parameter, it'll be ignored because the number of denoising steps is determined by the discrete timesteps the model is trained on and the declared fractional cutoff.

Let's set `denoising_end=0.8` so the base model performs the first 80% of denoising the **high-noise** timesteps and set `denoising_start=0.8` so the refiner model performs the last 20% of denoising the **low-noise** timesteps. The base model output should be in **latent** space instead of a PIL image.

```py
prompt = "A majestic lion jumping from a big stone at night"

image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
)[0]
image = refiner(
    prompt=prompt,
    num_inference_steps=40,
    denoising_start=0.8,
    image=image,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/90ddd439-c53a-4a13-b4d9-69d94be2fe95" alt="generated image of a lion on a rock at night" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">default base model</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/079d3421-2dab-4d8e-9ed3-0ce6aa2c4927" alt="generated image of a lion on a rock at night in higher quality" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">ensemble of expert denoisers</figcaption>
  </div>
</div>

The refiner model can also be used for inpainting in the [`StableDiffusionXLInpaintPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLInpaintPipeline):

```py
from mindone.diffusers import StableDiffusionXLInpaintPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import mindspore as ms

base = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    mindspore_dtype=ms.float16,
    use_safetensors=True,
)

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url)
mask_image = load_image(mask_url)

prompt = "A majestic tiger sitting on a bench"
num_inference_steps = 75
high_noise_frac = 0.7

image = base(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
)[0]
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
)[0][0]
make_image_grid([init_image, mask_image, image.resize((512, 512))], rows=1, cols=3)
```

This ensemble of expert denoisers method works well for all available schedulers!

### Base to refiner model

SDXL gets a boost in image quality by using the refiner model to add additional high-quality details to the fully-denoised image from the base model, in an image-to-image setting.

Load the base and refiner models:

```py
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    mindspore_dtype=ms.float16,
    use_safetensors=True,
)
```

!!! tip

    You can use SDXL refiner with a different base model. For example, you can use the [Hunyuan-DiT](../api/pipelines/hunyuandit.md) or [PixArt-Sigma](../api/pipelines/pixart_sigma.md) pipelines to generate images with better prompt adherence. Once you have generated an image, you can pass it to the SDXL refiner model to enhance final generation quality.

Generate an image from the base model, and set the model output to **latent** space:

```py
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

image = base(prompt=prompt, output_type="latent")[0][0]
```

Pass the generated image to the refiner model:

```py
image = refiner(prompt=prompt, image=image[None, :])[0][0]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/28bfa2d5-5cd1-47fa-a16f-740a2417f0de" alt="generated image of an astronaut riding a green horse on Mars" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">base model</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/6cce0e9d-b9b7-4570-8093-5463b0c1c389" alt="higher quality generated image of an astronaut riding a green horse on Mars" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">base model + refiner model</figcaption>
  </div>
</div>

For inpainting, load the base and the refiner model in the [`StableDiffusionXLInpaintPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLInpaintPipeline), remove the `denoising_end` and `denoising_start` parameters, and choose a smaller number of inference steps for the refiner.

## Micro-conditioning

SDXL training involves several additional conditioning techniques, which are referred to as *micro-conditioning*. These include original image size, target image size, and cropping parameters. The micro-conditionings can be used at inference time to create high-quality, centered images.

!!! tip

    You can use both micro-conditioning and negative micro-conditioning parameters thanks to classifier-free guidance. They are available in the [`StableDiffusionXLPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLPipeline), [`StableDiffusionXLImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLImg2ImgPipeline), [`StableDiffusionXLInpaintPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLInpaintPipeline), and [`StableDiffusionXLControlNetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/controlnet_sdxl/#mindone.diffusers.StableDiffusionXLControlNetPipeline).

### Size conditioning

There are two types of size conditioning:

- [`original_size`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLPipeline) conditioning comes from upscaled images in the training batch (because it would be wasteful to discard the smaller images which make up almost 40% of the total training data). This way, SDXL learns that upscaling artifacts are not supposed to be present in high-resolution images. During inference, you can use `original_size` to indicate the original image resolution. Using the default value of `(1024, 1024)` produces higher-quality images that resemble the 1024x1024 images in the dataset. If you choose to use a lower resolution, such as `(256, 256)`, the model still generates 1024x1024 images, but they'll look like the low resolution images (simpler patterns, blurring) in the dataset.

- [`target_size`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLPipeline) conditioning comes from finetuning SDXL to support different image aspect ratios. During inference, if you use the default value of `(1024, 1024)`, you'll get an image that resembles the composition of square images in the dataset. We recommend using the same value for `target_size` and `original_size`, but feel free to experiment with other options!

🤗 Diffusers also lets you specify negative conditions about an image's size to steer generation away from certain image resolutions:

```py
from mindone.diffusers import StableDiffusionXLPipeline
import mindspore as ms

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
    prompt=prompt,
    negative_original_size=(512, 512),
    negative_target_size=(1024, 1024),
)[0][0]
```

<div class="flex flex-col justify-center">
  <img src="https://github.com/user-attachments/assets/7994b92a-f33d-4149-bbb5-e517b06f9595"/>
  <figcaption class="text-center">Images negatively conditioned on image resolutions of (128, 128), (256, 256), and (512, 512).</figcaption>
</div>

### Crop conditioning

Images generated by previous Stable Diffusion models may sometimes appear to be cropped. This is because images are actually cropped during training so that all the images in a batch have the same size. By conditioning on crop coordinates, SDXL *learns* that no cropping - coordinates `(0, 0)` - usually correlates with centered subjects and complete faces (this is the default value in 🤗 Diffusers). You can experiment with different coordinates if you want to generate off-centered compositions!

```py
from mindone.diffusers import StableDiffusionXLPipeline
import mindspore as ms

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(prompt=prompt, crops_coords_top_left=(256, 0))[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/eca31309-17ad-4a9d-97d3-b925d6f5bcc8"/>
</div>

You can also specify negative cropping coordinates to steer generation away from certain cropping parameters:

```py
from mindone.diffusers import StableDiffusionXLPipeline
import mindspore as ms

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
    prompt=prompt,
    negative_original_size=(512, 512),
    negative_crops_coords_top_left=(0, 0),
    negative_target_size=(1024, 1024),
)[0][0]
image
```

## Use a different prompt for each text-encoder

SDXL uses two text-encoders, so it is possible to pass a different prompt to each text-encoder, which can [improve quality](https://github.com/huggingface/diffusers/issues/4004#issuecomment-1627764201). Pass your original prompt to `prompt` and the second prompt to `prompt_2` (use `negative_prompt` and `negative_prompt_2` if you're using negative prompts):

```py
from mindone.diffusers import StableDiffusionXLPipeline
import mindspore as ms

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
)

# prompt is passed to OAI CLIP-ViT/L-14
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt_2 is passed to OpenCLIP-ViT/bigG-14
prompt_2 = "Van Gogh painting"
image = pipeline(prompt=prompt, prompt_2=prompt_2)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/928ef371-6e04-4e40-8308-8333dc3d2d20" alt="generated image of an astronaut in a jungle in the style of a van gogh painting"/>
</div>

The dual text-encoders also support textual inversion embeddings that need to be loaded separately as explained in the [SDXL textual inversion](textual_inversion_inference.md#stable-diffusion-xl) section.

## Optimizations

SDXL is a large model, and you may need to optimize memory to get it to run on your hardware. Here is a tip to save memory and speed up inference.

Enable [xFormers](../optimization/xformers.md) to run SDXL:

```diff
+ base.enable_xformers_memory_efficient_attention()
+ refiner.enable_xformers_memory_efficient_attention()
```
