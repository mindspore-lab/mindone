<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quicktour

Diffusion models are trained to denoise random Gaussian noise step-by-step to generate a sample of interest, such as an image or audio. This has sparked a tremendous amount of interest in generative AI, and you have probably seen examples of diffusion generated images on the internet. 🧨 Diffusers is a library aimed at making diffusion models widely accessible to everyone.

Whether you're a developer or an everyday user, this quicktour will introduce you to 🧨 Diffusers and help you get up and generating quickly! There are three main components of the library to know about:

* The [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) is a high-level end-to-end class designed to rapidly generate samples from pretrained diffusion models for inference.
* Popular pretrained [model](./api/models/overview.md) architectures and modules that can be used as building blocks for creating diffusion systems.
* Many different [schedulers](./api/schedulers/overview.md) - algorithms that control how noise is added for training, and how to generate denoised images during inference.

The quicktour will show you how to use the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) for inference, and then walk you through how to combine a model and scheduler to replicate what's happening inside the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline).

!!! tip

    The quicktour is a simplified version of the introductory 🧨 Diffusers [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) to help you get started quickly. If you want to learn more about 🧨 Diffusers' goal, design philosophy, and additional details about its core API, check out the notebook!


Before you begin, make sure you have all the necessary libraries installed:

```py
# uncomment to install the necessary libraries in Colab
#!pip install --upgrade mindone transformers
```

- [🤗 Transformers](../transformers/index.md) is required to run the most popular diffusion models, such as [Stable Diffusion](./api/pipelines/stable_diffusion/overview.md).

## DiffusionPipeline

The [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) is the easiest way to use a pretrained diffusion system for inference. It is an end-to-end system containing the model and the scheduler. You can use the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) out-of-the-box for many tasks. Take a look at the table below for some supported tasks, and for a complete list of supported tasks, check out the [🧨 Diffusers Summary](./api/pipelines/overview.md#diffusers-summary) table.

| **Task**                               | **Description**                                                                                 | **Pipeline**                                                                          |
|----------------------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Unconditional Image Generation         | generate an image from Gaussian noise                                                           | [unconditional_image_generation](./using-diffusers/unconditional_image_generation.md) |
| Text-Guided Image Generation           | generate an image given a text prompt                                                           | [conditional_image_generation](./using-diffusers/conditional_image_generation.md)     |
| Text-Guided Image-to-Image Translation | adapt an image guided by a text prompt                                                          | [img2img](./using-diffusers/img2img.md)                                               |
| Text-Guided Image-Inpainting           | fill the masked part of an image given the image, the mask and a text prompt                    | [inpaint](./using-diffusers/inpaint.md)                                               |
| Text-Guided Depth-to-Image Translation | adapt parts of an image guided by a text prompt while preserving structure via depth estimation | [depth2img](./using-diffusers/depth2img.md)                                           |

Start by creating an instance of a [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) and specify which pipeline checkpoint you would like to download.
You can use the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) for any [checkpoint](https://huggingface.co/models?library=diffusers&sort=downloads) stored on the Hugging Face Hub.
In this quicktour, you'll load the [`stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) checkpoint for text-to-image generation.

!!! warning

    For [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion) models, please carefully read the [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license) first before running the model. 🧨 Diffusers implements a [`safety_checker`](https://github.com/The-truthh/mindone/blob/docs/mindone/diffusers/pipelines/stable_diffusion/safety_checker.py) to prevent offensive or harmful content, but the model's improved image generation capabilities can still produce potentially harmful content.


Load the model with the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) method:

!!! tip

    MindONE.diffusers currently does not support loading `.bin` files, if the models in the [Hub](https://huggingface.co/models) consist solely of `.bin` files, please refer to the [`tutorial`](using-diffusers/other-formats.md#bin-files)

    If the connection error occurs while loading the weights, try configuring the [`HF_ENDPOINT`](https://huggingface.co/docs/huggingface_hub/v0.13.2/package_reference/environment_variables#hfendpoint) environment variable to switch to an alternative mirror.

```diff
- from diffusers import DiffusionPipeline
+ from mindone.diffusers import DiffusionPipeline

  pipeline = DiffusionPipeline.from_pretrained(
       "stable-diffusion-v1-5/stable-diffusion-v1-5",
-      torch_dtype=torch.float32,
+      mindspore_dtype=mindspore.float32,
       use_safetensors=True
  )
```

The [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) downloads and caches all modeling, tokenization, and scheduling components. You'll see that the Stable Diffusion pipeline is composed of the [`UNet2DConditionModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d-cond/#mindone.diffusers.UNet2DConditionModel) and [`PNDMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/pndm/#mindone.diffusers.PNDMScheduler) among other things:

```python
pipeline
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.21.4",
  ...,
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  ...,
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

We strongly recommend running the pipeline on a NPU because the model consists of roughly 1.4 billion parameters. You can't move the generator object to a NPU **manually**, because MindSpore implicitly does that. Do **NOT** invoke `to("cuda")`:

```diff
- pipeline.to("cuda")
```

Now you can pass a text prompt to the `pipeline` to generate an image, and then access the denoised image. By default, the image output is wrapped in a [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) object.

```python
image = pipeline("An image of a squirrel in Picasso style")[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/55998cd5-5657-4efa-bb35-d70c9ab96dde"/>
</div>

Save the image by calling `save`:

```python
image.save("image_of_squirrel_painting.png")
```

### Local pipeline

You can also use the pipeline locally. The only difference is you need to download the weights first:

```bash
!git lfs install
!git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

Then load the saved weights into the pipeline:

```python
pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

Now, you can run the pipeline as you would in the section above.

### Swapping schedulers

Different schedulers come with different denoising speeds and quality trade-offs. The best way to find out which one works best for you is to try them out! One of the main features of 🧨 Diffusers is to allow you to easily switch between schedulers. For example, to replace the default [`PNDMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/pndm/#mindone.diffusers.PNDMScheduler) with the [`EulerDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/euler/#mindone.diffusers.EulerDiscreteScheduler), load it with the [`from_config`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/configuration/#mindone.diffusers.configuration_utils.ConfigMixin.from_config) method:

```python
from mindone.diffusers import EulerDiscreteScheduler

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```

Try generating an image with the new scheduler and see if you notice a difference!

In the next section, you'll take a closer look at the components - the model and scheduler - that make up the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) and learn how to use these components to generate an image of a cat.

## Models

Most models take a noisy sample, and at each timestep it predicts the *noise residual* (other models learn to predict the previous sample directly or the velocity or [`v-prediction`](https://github.com/The-truthh/mindone/blob/docs/mindone/diffusers/schedulers/scheduling_ddim.py#L162)), the difference between a less noisy image and the input image. You can mix and match models to create other diffusion systems.

Models are initiated with the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin.from_pretrained) method which also locally caches the model weights so it is faster the next time you load the model. For the quicktour, you'll load the [`UNet2DModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d/#mindone.diffusers.UNet2DModel), a basic unconditional image generation model with a checkpoint trained on cat images:

```python
from mindone.diffusers import UNet2DModel

repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)
```

To access the model parameters, call `model.config`:

```python
model.config
```

The model configuration is a 🧊 frozen 🧊 dictionary, which means those parameters can't be changed after the model is created. This is intentional and ensures that the parameters used to define the model architecture at the start remain the same, while other parameters can still be adjusted during inference.

Some of the most important parameters are:

* `sample_size`: the height and width dimension of the input sample.
* `in_channels`: the number of input channels of the input sample.
* `down_block_types` and `up_block_types`: the type of down- and upsampling blocks used to create the UNet architecture.
* `block_out_channels`: the number of output channels of the downsampling blocks; also used in reverse order for the number of input channels of the upsampling blocks.
* `layers_per_block`: the number of ResNet blocks present in each UNet block.

To use the model for inference, create the image shape with random Gaussian noise. It should have a `batch` axis because the model can receive multiple random noises, a `channel` axis corresponding to the number of input channels, and a `sample_size` axis for the height and width of the image:

```python
import mindspore

noisy_sample = mindspore.ops.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
noisy_sample.shape
[1, 3, 256, 256]
```

For inference, pass the noisy image and a `timestep` to the model. The `timestep` indicates how noisy the input image is, with more noise at the beginning and less at the end. This helps the model determine its position in the diffusion process, whether it is closer to the start or the end. Use the `sample` method to get the model output:

```python
noisy_residual = model(sample=noisy_sample, timestep=2)[0]
```

To generate actual examples though, you'll need a scheduler to guide the denoising process. In the next section, you'll learn how to couple a model with a scheduler.

## Schedulers

Schedulers manage going from a noisy sample to a less noisy sample given the model output - in this case, it is the `noisy_residual`.

!!! tip

    🧨 Diffusers is a toolbox for building diffusion systems. While the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) is a convenient way to get started with a pre-built diffusion system, you can also choose your own model and scheduler components separately to build a custom diffusion system.


For the quicktour, you'll instantiate the [`DDPMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddpm/#mindone.diffusers.DDPMScheduler) with its [`from_config`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/configuration/#mindone.diffusers.configuration_utils.ConfigMixin.from_config) method:

```python
from mindone.diffusers import DDPMScheduler

scheduler = DDPMScheduler.from_pretrained(repo_id)
scheduler
DDPMScheduler {
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.21.4",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "clip_sample": true,
  "clip_sample_range": 1.0,
  "dynamic_thresholding_ratio": 0.995,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "sample_max_value": 1.0,
  "steps_offset": 0,
  "thresholding": false,
  "timestep_spacing": "leading",
  "trained_betas": null,
  "variance_type": "fixed_small"
}
```

!!! tip

    💡 Unlike a model, a scheduler does not have trainable weights and is parameter-free!


Some of the most important parameters are:

* `num_train_timesteps`: the length of the denoising process or, in other words, the number of timesteps required to process random Gaussian noise into a data sample.
* `beta_schedule`: the type of noise schedule to use for inference and training.
* `beta_start` and `beta_end`: the start and end noise values for the noise schedule.

To predict a slightly less noisy image, pass the following to the scheduler's [`step`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddpm/#mindone.diffusers.DDPMScheduler.step) method: model output, `timestep`, and current `sample`.

```python
less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample)[0]
less_noisy_sample.shape
[1, 3, 256, 256]
```

The `less_noisy_sample` can be passed to the next `timestep` where it'll get even less noisy! Let's bring it all together now and visualize the entire denoising process.

First, create a function that postprocesses and displays the denoised image as a `PIL.Image`:

```python
import PIL.Image
import numpy as np


def display_sample(sample, i):
    image_processed = sample.permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)
```

Now create a denoising loop that predicts the residual of the less noisy sample, and computes the less noisy sample with the scheduler:

```python
import tqdm

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    residual = model(sample, t)[0]

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample)[0]

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        display_sample(sample, i + 1)
```

Sit back and watch as a cat is generated from nothing but noise! 😻

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/b6a46f48-4c86-4687-8bb0-216f21f63c3b"/>
</div>

## Next steps

Hopefully, you generated some cool images with 🧨 Diffusers in this quicktour! For your next steps, you can:

* Train or finetune a model to generate your own images in the [training](./tutorials/basic_training.md) tutorial.
* See example official and community [training or finetuning scripts](https://github.com/The-truthh/mindone/tree/docs/examples/diffusers) for a variety of use cases.
* Learn more about loading, accessing, changing, and comparing schedulers in the [Using different Schedulers](./using-diffusers/schedulers.md) guide.
* Explore prompt engineering, speed and memory optimizations, and tips and tricks for generating higher-quality images with the [Stable Diffusion](./stable_diffusion.md) guide.
* Dive deeper into speeding up 🧨 Diffusers with guides on [optimized MindSpore on a NPU](./optimization/fp16.md).
