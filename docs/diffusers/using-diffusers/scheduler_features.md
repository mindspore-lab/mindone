<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Scheduler features

The scheduler is an important component of any diffusion model because it controls the entire denoising (or sampling) process. There are many types of schedulers, some are optimized for speed and some for quality. With Diffusers, you can modify the scheduler configuration to use custom noise schedules, sigmas, and rescale the noise schedule. Changing these parameters can have profound effects on inference quality and speed.

This guide will demonstrate how to use these features to improve inference quality.

!!! tip

    Diffusers currently only supports the `timesteps` and `sigmas` parameters for a select list of schedulers and pipelines.

## Timestep schedules

The timestep or noise schedule determines the amount of noise at each sampling step. The scheduler uses this to generate an image with the corresponding amount of noise at each step. The timestep schedule is generated from the scheduler's default configuration, but you can customize the scheduler to use new and optimized sampling schedules that aren't in Diffusers yet.

For example, [Align Your Steps (AYS)](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/) is a method for optimizing a sampling schedule to generate a high-quality image in as little as 10 steps. The optimal `10-step schedule` for Stable Diffusion XL is:

```py
sampling_schedule = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]
```

You can use the AYS sampling schedule in a pipeline by passing it to the `timesteps` parameter.

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    mindspore_dtype=ms.float16,
    variant="fp16",
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++")

prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
generator = np.random.Generator(np.random.PCG64(2487854446))
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    generator=generator,
    timesteps=sampling_schedule,
)[0][0]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/8a97dc6f-c025-4801-b873-a17eae7082e0"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">AYS timestep schedule 10 steps</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/690edebd-072d-4cd7-8d20-ed60abb2b89a"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Linearly-spaced timestep schedule 10 steps</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c4c96e2f-7cad-40d4-9b49-4068dc1f9344"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Linearly-spaced timestep schedule 25 steps</figcaption>
  </div>
</div>

## Timestep spacing

The way sample steps are selected in the schedule can affect the quality of the generated image, especially with respect to [rescaling the noise schedule](#rescale-noise-schedule), which can enable a model to generate much brighter or darker images. Diffusers provides three timestep spacing methods:

- `leading` creates evenly spaced steps
- `linspace` includes the first and last steps and evenly selects the remaining intermediate steps
- `trailing` only includes the last step and evenly selects the remaining intermediate steps starting from the end

It is recommended to use the `trailing` spacing method because it generates higher quality images with more details when there are fewer sample steps. But the difference in quality is not as obvious for more standard sample step values.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import numpy as np

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    mindspore_dtype=ms.float16,
    variant="fp16",
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")

prompt = "A cinematic shot of a cute little black cat sitting on a pumpkin at night"
generator = np.random.Generator(np.random.PCG64(2487854446))
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    generator=generator,
    num_inference_steps=5,
)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/4e4ba7e5-d85f-4e85-9244-341bb1feae00"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">trailing spacing after 5 steps</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/9f75e827-ed3d-4b55-b3b7-aa999af52bf3"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">leading spacing after 5 steps</figcaption>
  </div>
</div>

## Sigmas

The `sigmas` parameter is the amount of noise added at each timestep according to the timestep schedule. Like the `timesteps` parameter, you can customize the `sigmas` parameter to control how much noise is added at each step. When you use a custom `sigmas` value, the `timesteps` are calculated from the custom `sigmas` value and the default scheduler configuration is ignored.

For example, you can manually pass the `sigmas` for something like the 10-step AYS schedule from before to the pipeline.

```py
import mindspore as ms
import numpy as np

from mindone.diffusers import DiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  mindspore_dtype=ms.float16,
)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.0]
prompt = "anthropomorphic capybara wearing a suit and working with a computer"
generator = np.random.Generator(np.random.PCG64(123))
image = pipeline(
    prompt=prompt,
    num_inference_steps=10,
    sigmas=sigmas,
    generator=generator
)[0][0]
```

When you take a look at the scheduler's `timesteps` parameter, you'll see that it is the same as the AYS timestep schedule because the `timestep` schedule is calculated from the `sigmas`.

```py
print(f" timesteps: {pipeline.scheduler.timesteps}")
"timesteps: [999., 845., 730., 587., 443., 310., 193., 116.,  53.,  13.]"
```

### Karras sigmas

!!! tip

    Refer to the scheduler API [overview](../api/schedulers/overview.md) for a list of schedulers that support Karras sigmas.

    Karras sigmas should not be used for models that weren't trained with them. For example, the base Stable Diffusion XL model shouldn't use Karras sigmas but the [DreamShaperXL](https://hf.co/Lykon/dreamshaper-xl-1-0) model can since they are trained with Karras sigmas.

Karras scheduler's use the timestep schedule and sigmas from the [Elucidating the Design Space of Diffusion-Based Generative Models](https://hf.co/papers/2206.00364) paper. This scheduler variant applies a smaller amount of noise per step as it approaches the end of the sampling process compared to other schedulers, and can increase the level of details in the generated image.

Enable Karras sigmas by setting `use_karras_sigmas=True` in the scheduler.

```py
import mindspore as ms
from mindone.diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import numpy as np

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    mindspore_dtype=ms.float16,
    variant="fp16",
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)

prompt = "A cinematic shot of a cute little rabbit wearing a jacket and doing a thumbs up"
generator = np.random.Generator(np.random.PCG64(2487854446))
image = pipeline(
    prompt=prompt,
    negative_prompt="",
    generator=generator,
)[0][0]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/227258b6-61d3-4b83-84ad-833699d6a31e"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Karras sigmas enabled</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/75ad479a-ae02-4c1a-ab82-c968e12b0138"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Karras sigmas disabled</figcaption>
  </div>
</div>

## Rescale noise schedule

In the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://hf.co/papers/2305.08891) paper, the authors discovered that common noise schedules allowed some signal to leak into the last timestep. This signal leakage at inference can cause models to only generate images with medium brightness. By enforcing a zero signal-to-noise ratio (SNR) for the timstep schedule and sampling from the last timestep, the model can be improved to generate very bright or dark images.

!!! tip

    For inference, you need a model that has been trained with *v_prediction*. To train your own model with *v_prediction*, add the following flag to the [train_text_to_image.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image.py) or [train_text_to_image_lora.py](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/text_to_image/train_text_to_image_lora.py) scripts.

    ```bash
    --prediction_type="v_prediction"
    ```

For example, load the [ptx0/pseudo-journey-v2](https://hf.co/ptx0/pseudo-journey-v2) checkpoint which was trained with `v_prediction` and the [`DDIMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddim/#mindone.diffusers.DDIMScheduler). Configure the following parameters in the [`DDIMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddim/#mindone.diffusers.DDIMScheduler):

* `rescale_betas_zero_snr=True` to rescale the noise schedule to zero SNR
* `timestep_spacing="trailing"` to start sampling from the last timestep

Set `guidance_rescale` in the pipeline to prevent over-exposure. A lower value increases brightness but some of the details may appear washed out.

```py
from mindone.diffusers import DiffusionPipeline, DDIMScheduler
import numpy as np

pipeline = DiffusionPipeline.from_pretrained("ptx0/pseudo-journey-v2", use_safetensors=True)

pipeline.scheduler = DDIMScheduler.from_config(
    pipeline.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)
prompt = "cinematic photo of a snowy mountain at night with the northern lights aurora borealis overhead, 35mm photograph, film, professional, 4k, highly detailed"
generator = np.random.Generator(np.random.PCG64(23))
image = pipeline(prompt, guidance_rescale=0.7, generator=generator)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/037a4c53-02bb-4901-a3b6-95dca1fbd48d"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">default Stable Diffusion v2-1 image</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c3f041fb-8be6-48ee-b9a4-83b18954ff40"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">image with zero SNR and trailing timestep spacing enabled</figcaption>
  </div>
</div>
