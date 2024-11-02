<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load schedulers and models

Diffusion pipelines are a collection of interchangeable schedulers and models that can be mixed and matched to tailor a pipeline to a specific use case. The scheduler encapsulates the entire denoising process such as the number of denoising steps and the algorithm for finding the denoised sample. A scheduler is not parameterized or trained so they don't take very much memory. The model is usually only concerned with the forward pass of going from a noisy input to a less noisy sample.

This guide will show you how to load schedulers and models to customize a pipeline. You'll use the [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) checkpoint throughout this guide, so let's load it first.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, use_safetensors=True
)
```

You can see what scheduler this pipeline uses with the `pipeline.scheduler` attribute.

```py
pipeline.scheduler
PNDMScheduler {
  "_class_name": "PNDMScheduler",
  "_diffusers_version": "0.29.2",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "leading",
  "trained_betas": null
}
```

## Load a scheduler

Schedulers are defined by a configuration file that can be used by a variety of schedulers. Load a scheduler with the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/overview/#mindone.diffusers.SchedulerMixin.from_pretrained) method, and specify the `subfolder` parameter to load the configuration file into the correct subfolder of the pipeline repository.

For example, to load the [`DDIMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddim/#mindone.diffusers.DDIMScheduler):

```py
from mindone.diffusers import DDIMScheduler, DiffusionPipeline

ddim = DDIMScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
```

Then you can pass the newly loaded scheduler to the pipeline.

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", scheduler=ddim, mindspore_dtype=ms.float16, use_safetensors=True
)
```

## Compare schedulers

Schedulers have their own unique strengths and weaknesses, making it difficult to quantitatively compare which scheduler works best for a pipeline. You typically have to make a trade-off between denoising speed and denoising quality. We recommend trying out different schedulers to find one that works best for your use case. Call the `pipeline.scheduler.compatibles` attribute to see what schedulers are compatible with a pipeline.

Let's compare the [`LMSDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/lms_discrete/#mindone.diffusers.LMSDiscreteScheduler), [`EulerDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/euler/#mindone.diffusers.EulerDiscreteScheduler), [`EulerAncestralDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/euler_ancestral/#mindone.diffusers.EulerAncestralDiscreteScheduler), and the [`DPMSolverMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/multistep_dpm_solver/#mindone.diffusers.DPMSolverMultistepScheduler) on the following prompt and seed.

```py
import mindspore as ms
from mindone.diffusers import DiffusionPipeline
import numpy as np

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16, use_safetensors=True
)

prompt = "A photograph of an astronaut riding a horse on Mars, high resolution, high definition."
generator = np.random.Generator(np.random.PCG64(8))
```

To change the pipelines scheduler, use the [`from_config`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/configuration/#mindone.diffusers.configuration_utils.ConfigMixin.from_config) method to load a different scheduler's `pipeline.scheduler.config` into the pipeline.

=== "LMSDiscreteScheduler"

    [`LMSDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/lms_discrete/#mindone.diffusers.LMSDiscreteScheduler) typically generates higher quality images than the default scheduler.

    ```py
    from mindone.diffusers import LMSDiscreteScheduler

    pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    image = pipeline(prompt, generator=generator)[0][0]
    image
    ```

=== "EulerDiscreteScheduler"

    [`EulerDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/euler/#mindone.diffusers.EulerDiscreteScheduler) can generate higher quality images in just 30 steps.

    ```py
    from mindone.diffusers import EulerDiscreteScheduler

    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    image = pipeline(prompt, generator=generator)[0][0]
    image
    ```

=== "EulerAncestralDiscreteScheduler"

    [`EulerAncestralDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/euler_ancestral/#mindone.diffusers.EulerAncestralDiscreteScheduler) can generate higher quality images in just 30 steps.

    ```py
    from mindone.diffusers import EulerAncestralDiscreteScheduler

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    image = pipeline(prompt, generator=generator)[0][0]
    image
    ```

=== "DPMSolverMultistepScheduler"

    [`DPMSolverMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/multistep_dpm_solver/#mindone.diffusers.DPMSolverMultistepScheduler) provides a balance between speed and quality and can generate higher quality images in just 20 steps.

    ```py
    from mindone.diffusers import DPMSolverMultistepScheduler

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    image = pipeline(prompt, generator=generator)[0][0]
    image
    ```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/f0f97746-a72c-4638-a0a4-a92836378f4f" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">LMSDiscreteScheduler</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/f293240a-1a9d-47e0-8b3c-13410211029f" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">EulerDiscreteScheduler</figcaption>
  </div>
</div>
<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/5235dd0c-4ded-46bf-9d83-d18a377c4785" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">EulerAncestralDiscreteScheduler</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/f6774dc4-66a4-49c4-8a87-b541641f14e8" />
    <figcaption class="mt-2 text-center text-sm text-gray-500">DPMSolverMultistepScheduler</figcaption>
  </div>
</div>

Most images look very similar and are comparable in quality. Again, it often comes down to your specific use case so a good approach is to run multiple different schedulers and compare the results.

## Models

Models are loaded from the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin.from_pretrained) method, which downloads and caches the latest version of the model weights and configurations. If the latest files are available in the local cache, [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin.from_pretrained) reuses files in the cache instead of re-downloading them.

Models can be loaded from a subfolder with the `subfolder` argument. For example, the model weights for [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) are stored in the [unet](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet) subfolder.

```python
from mindone.diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", use_safetensors=True)
```

They can also be directly loaded from a [repository](https://huggingface.co/google/ddpm-cifar10-32/tree/main).

```python
from mindone.diffusers import UNet2DModel

unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
```

To load and save model variants, specify the `variant` argument in [`ModelMixin.from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin.from_pretrained) and [`ModelMixin.save_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin.save_pretrained).

```python
from mindone.diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant="non_ema", use_safetensors=True
)
unet.save_pretrained("./local-unet", variant="non_ema")
```
