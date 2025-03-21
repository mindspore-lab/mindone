<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Flux

Flux is a series of text-to-image generation models based on diffusion transformers. To know more about Flux, check out the original [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/) by the creators of Flux, Black Forest Labs.

Original model checkpoints for Flux can be found [here](https://huggingface.co/black-forest-labs). Original inference code can be found [here](https://github.com/black-forest-labs/flux).

Flux comes in two variants:
| model type | model id |
|:----------:|:--------:|
| Timestep-distilled | [`black-forest-labs/FLUX.1-schnell`](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| Guidance-distilled | [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Fill Inpainting/Outpainting (Guidance-distilled) | [`black-forest-labs/FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) |
| Canny Control (Guidance-distilled) | [`black-forest-labs/FLUX.1-Canny-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) |
| Depth Control (Guidance-distilled) | [`black-forest-labs/FLUX.1-Depth-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) |
| Redux | [`black-forest-labs/FLUX.1-Redux-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) |

All checkpoints have different usage which we detail below.

### Timestep-distilled

* `max_sequence_length` cannot be more than 256.
* `guidance_scale` needs to be 0.
* As this is a timestep-distilled model, it benefits from fewer sampling steps.

```python
import mindspore
from mindone.diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", mindspore_dtype=mindspore.bfloat16)

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
)[0][0]
out.save("image.png")
```

### Guidance-distilled

* The guidance-distilled variant takes about 50 sampling steps for good-quality generation.
* It doesn't have any limitations around the `max_sequence_length`.

```python
import mindspore
from mindone.diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", mindspore_dtype=mindspore.bfloat16)

prompt = "a tiny astronaut hatching from an egg on the moon"
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=768,
    width=1360,
    num_inference_steps=50,
)[0][0]
out.save("image.png")
```

### Fill Inpainting/Outpainting

* Flux Fill pipeline does not require strength as an input like regular inpainting pipelines.
* It supports both inpainting and outpainting.

```python
import mindspore as ms
import numpy as np
from mindone.diffusers import FluxFillPipeline
from mindone.diffusers.utils import load_image

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/cup_mask.png")

repo_id = "black-forest-labs/FLUX.1-Fill-dev"
pipe = FluxFillPipeline.from_pretrained(repo_id, mindspore_dtype=ms.bfloat16)

image = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    max_sequence_length=512,
    generator=np.random.Generator(np.random.PCG64(0))
)[0][0]
image.save(f"output.png")
```

### Canny Control

**Note:** `black-forest-labs/Flux.1-Canny-dev` is _not_ a [`ControlNetModel`] model. ControlNet models are a separate component from the UNet/Transformer whose residuals are added to the actual underlying model. Canny Control is an alternate architecture that achieves effectively the same results as a ControlNet model would, by using channel-wise concatenation with input control condition and ensuring the transformer learns structure control by following the condition as closely as possible.

!!! warning

    ⚠️ MindONE currently does not support the full process for the control image generating, as MindONE does not yet support `CannyDetector` from controlnet_aux. Therefore, you need to prepare the `control_image` in advance to continue the process.

```python
# !pip install -U controlnet-aux
import mindspore as ms
# from controlnet_aux import CannyDetector
from mindone.diffusers import FluxControlPipeline
from mindone.diffusers.utils import load_image

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", mindspore_dtype=ms.bfloat16)

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."

control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")
# processor = CannyDetector()
# control_image = processor(control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
control_image = load_image("path/to/control_image")

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
)[0][0]
image.save("output.png")
```

### Depth Control

**Note:** `black-forest-labs/Flux.1-Depth-dev` is _not_ a ControlNet model. [`ControlNetModel`] models are a separate component from the UNet/Transformer whose residuals are added to the actual underlying model. Depth Control is an alternate architecture that achieves effectively the same results as a ControlNet model would, by using channel-wise concatenation with input control condition and ensuring the transformer learns structure control by following the condition as closely as possible.

!!! warning

    ⚠️ MindONE currently does not support the full process for the control image generating, as MindONE does not yet support `DepthPreprocessor` from image_gen_aux. Therefore, you need to prepare the `control_image` in advance to continue the process.

```python
# !pip install git+https://github.com/huggingface/image_gen_aux
import mindspore as ms
import numpy as np
from mindone.diffusers import FluxControlPipeline, FluxTransformer2DModel
from mindone.diffusers.utils import load_image
# from image_gen_aux import DepthPreprocessor

pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", mindspore_dtype=ms.bfloat16)

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

# processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
# control_image = processor(control_image)[0].convert("RGB")
control_image = load_image("path/to/control_image")

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=10.0,
    generator=np.random.Generator(np.random.PCG64(0)),
)[0][0]
image.save("output.png")
```

### Redux

* Flux Redux pipeline is an adapter for FLUX.1 base models. It can be used with both flux-dev and flux-schnell, for image-to-image generation.
* You can first use the `FluxPriorReduxPipeline` to get the `prompt_embeds` and `pooled_prompt_embeds`, and then feed them into the `FluxPipeline` for image-to-image generation.
* When use `FluxPriorReduxPipeline` with a base pipeline, you can set `text_encoder=None` and `text_encoder_2=None` in the base pipeline, in order to save VRAM.

```python
import mindspore as ms
import numpy as np
from mindone.diffusers import FluxPriorReduxPipeline, FluxPipeline
from mindone.diffusers.utils import load_image
dtype = ms.bfloat16

repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
repo_base = "black-forest-labs/FLUX.1-dev"
pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, mindspore_dtype=dtype)
pipe = FluxPipeline.from_pretrained(
    repo_base,
    text_encoder=None,
    text_encoder_2=None,
    mindspore_dtype=ms.bfloat16
)

image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy/img5.png")
prompt_embeds, pooled_prompt_embeds = pipe_prior_redux(image)
images = pipe(
    guidance_scale=2.5,
    num_inference_steps=50,
    generator=np.random.Generator(np.random.PCG64(0)),
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
)[0]
images[0].save("flux-redux.png")
```

## Running FP16 inference
Flux can generate high-quality images with FP16 but produces different outputs compared to FP32/BF16. The issue is that some activations in the text encoders have to be clipped when running in FP16, which affects the overall image. Forcing text encoders to run with FP32 inference thus removes this output difference. See [here](https://github.com/huggingface/diffusers/pull/9097#issuecomment-2272292516) for details.

FP16 inference code:
```python
import mindspore
from mindone.diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", mindspore_dtype=mindspore.bfloat16) # can replace schnell with dev
# to run on low vram devices
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(mindspore.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

prompt = "A cat holding a sign that says hello world"
out = pipe(
    prompt=prompt,
    guidance_scale=0.,
    height=768,
    width=1360,
    num_inference_steps=4,
    max_sequence_length=256,
)[0][0]
out.save("image.png")
```

## Single File Loading for the `FluxTransformer2DModel`

The `FluxTransformer2DModel` supports loading checkpoints in the original format shipped by Black Forest Labs. This is also useful when trying to load finetunes or quantized versions of the models that have been published by the community.


```python
import numpy as np

import mindspore
from mindone.diffusers import FluxTransformer2DModel, FluxPipeline
from mindone.transformers import T5EncoderModel, CLIPTextModel

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = mindspore.bfloat16

transformer = FluxTransformer2DModel.from_single_file("https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", mindspore_dtype=dtype)

text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", mindspore_dtype=dtype)

pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, mindspore_dtype=dtype)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=3.5,
    output_type="pil",
    num_inference_steps=20,
    generator=np.random.Generator(np.random.PCG64(0))
)[0][0]

image.save("flux.png")
```


::: mindone.diffusers.FluxPipeline

::: mindone.diffusers.FluxImg2ImgPipeline

::: mindone.diffusers.FluxInpaintPipeline

::: mindone.diffusers.FluxControlNetInpaintPipeline

::: mindone.diffusers.FluxControlNetImg2ImgPipeline

::: mindone.diffusers.FluxControlPipeline

::: mindone.diffusers.FluxControlImg2ImgPipeline

::: mindone.diffusers.FluxPriorReduxPipeline

::: mindone.diffusers.FluxFillPipeline
