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

* Timestep-distilled (`black-forest-labs/FLUX.1-schnell`)
* Guidance-distilled (`black-forest-labs/FLUX.1-dev`)

Both checkpoints have slightly difference usage which we detail below.

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


::: mindone.diffusers.pipelines.flux.FluxPipeline
