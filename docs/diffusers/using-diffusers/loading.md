<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Load pipelines

Diffusion systems consist of multiple components like parameterized models and schedulers that interact in complex ways. That is why we designed the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) to wrap the complexity of the entire diffusion system into an easy-to-use API. At the same time, the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) is entirely customizable so you can modify each component to build a diffusion system for your use case.

This guide will show you how to load:

- pipelines from the Hub and locally
- different components into a pipeline
- multiple pipelines without increasing memory usage
- checkpoint variants such as different floating point types or non-exponential mean averaged (EMA) weights

## Load a pipeline

!!! tip

    Skip to the [DiffusionPipeline explained](#diffusionpipeline-explained) section if you're interested in an explanation about how the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) class works.

There are two ways to load a pipeline for a task:

1. Load the generic [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) class and allow it to automatically detect the correct pipeline class from the checkpoint.
2. Load a specific pipeline class for a specific task.

=== "Generic pipeline"

    The [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) class is a simple and generic way to load the latest trending diffusion model from the [Hub](https://huggingface.co/models?library=diffusers&sort=trending). It uses the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) method to automatically detect the correct pipeline class for a task from the checkpoint, downloads and caches all the required configuration and weight files, and returns a pipeline ready for inference.

    ```python
    from mindone.diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
    ```

    This same checkpoint can also be used for an image-to-image task. The [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) class can handle any task as long as you provide the appropriate inputs. For example, for an image-to-image task, you need to pass an initial image to the pipeline.

    ```py
    from mindone.diffusers import DiffusionPipeline
    from mindone.diffusers.utils import load_image

    pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)

    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=init_image)[0][0]
    ```

=== "Specific pipeline"

    Checkpoints can be loaded by their specific pipeline class if you already know it. For example, to load a Stable Diffusion model, use the [`StableDiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/text2img/#mindone.diffusers.StableDiffusionPipeline) class.

    ```python
    from mindone.diffusers import StableDiffusionPipeline

    pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
    ```

    This same checkpoint may also be used for another task like image-to-image. To differentiate what task you want to use the checkpoint for, you have to use the corresponding task-specific pipeline class. For example, to use the same checkpoint for image-to-image, use the [`StableDiffusionImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/img2img/#mindone.diffusers.StableDiffusionImg2ImgPipeline) class.

    ```py
    from mindone.diffusers import StableDiffusionImg2ImgPipeline

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)
    ```

### Local pipeline

To load a pipeline locally, use [git-lfs](https://git-lfs.github.com/) to manually download a checkpoint to your local disk.

```bash
git-lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

This creates a local folder, ./stable-diffusion-v1-5, on your disk and you should pass its path to [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained).

```python
from mindone.diffusers import DiffusionPipeline

stable_diffusion = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", use_safetensors=True)
```

The [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) method won't download files from the Hub when it detects a local path, but this also means it won't download and cache the latest changes to a checkpoint.

## Customize a pipeline

You can customize a pipeline by loading different components into it. This is important because you can:

- change to a scheduler with faster generation speed or higher generation quality depending on your needs (call the `scheduler.compatibles` method on your pipeline to see compatible schedulers)
- change a default pipeline component to a newer and better performing one

For example, let's customize the default [stabilityai/stable-diffusion-xl-base-1.0](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0) checkpoint with:

- The [`HeunDiscreteScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/heun/#mindone.diffusers.HeunDiscreteScheduler) to generate higher quality images at the expense of slower generation speed. You must pass the `subfolder="scheduler"` parameter in [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/heun/#mindone.diffusers.HeunDiscreteScheduler.from_pretrained) to load the scheduler configuration into the correct [subfolder](https://hf.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main/scheduler) of the pipeline repository.
- A more stable VAE that runs in fp16.

```py
from mindone.diffusers import StableDiffusionXLPipeline, HeunDiscreteScheduler, AutoencoderKL
import mindspore as ms

scheduler = HeunDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms.float16, use_safetensors=True)
```

Now pass the new scheduler and VAE to the [`StableDiffusionXLPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLPipeline).

```py
pipeline = StableDiffusionXLPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0",
  scheduler=scheduler,
  vae=vae,
  mindspore_dtype=ms.float16,
  use_safetensors=True
)
```

## Safety checker

Diffusers implements a [safety checker](https://github.com/The-truthh/mindone/blob/docs/mindone/diffusers/pipelines/stable_diffusion/safety_checker.py) for Stable Diffusion models which can generate harmful content. The safety checker screens the generated output against known hardcoded not-safe-for-work (NSFW) content. If for whatever reason you'd like to disable the safety checker, pass `safety_checker=None` to the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) method.

```python
from mindone.diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None, use_safetensors=True)
"""
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide by the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend keeping the safety filter enabled in all public-facing circumstances, disabling it only for use cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
"""
```

## Checkpoint variants

A checkpoint variant is usually a checkpoint whose weights are:

- Stored in a different floating point type, such as [mindspore.float16](https://www.mindspore.cn/docs/zh-CN/r2.3.1/api_python/mindspore/mindspore.dtype.html#mindspore.dtype), because it only requires half the bandwidth and storage to download. You can't use this variant if you're continuing training or using a CPU.
- Non-exponential mean averaged (EMA) weights which shouldn't be used for inference. You should use this variant to continue finetuning a model.

!!! tip

    When the checkpoints have identical model structures, but they were trained on different datasets and with a different training setup, they should be stored in separate repositories. For example, [stabilityai/stable-diffusion-2](https://hf.co/stabilityai/stable-diffusion-2) and [stabilityai/stable-diffusion-2-1](https://hf.co/stabilityai/stable-diffusion-2-1) are stored in separate repositories.

Otherwise, a variant is **identical** to the original checkpoint. They have exactly the same serialization format (like [safetensors](https://huggingface.co/docs/diffusers/main/en/using-diffusers/using_safetensors)), model structure, and their weights have identical tensor shapes.

| **checkpoint type** | **weight name**                      | **argument for loading weights** |
|---------------------|--------------------------------------|----------------------------------|
| original            | diffusion_model.safetensors          |                                  |
| floating point      | diffusion_model.fp16.safetensors     | `variant`, `mindspore_dtype`     |
| non-EMA             | diffusion_model.non_ema.safetensors  | `variant`                        |

There are two important arguments for loading variants:

- `mindspore_dtype` specifies the floating point precision of the loaded checkpoint. For example, if you want to save bandwidth by loading a fp16 variant, you should set `variant="fp16"` and `mindspore_dtype=mindspore.float16` to *convert the weights* to fp16. Otherwise, the fp16 weights are converted to the default fp32 precision.

  If you only set `mindspore_dtype=mindspore.float16`, the default fp32 weights are downloaded first and then converted to fp16.

- `variant` specifies which files should be loaded from the repository. For example, if you want to load a non-EMA variant of a UNet from [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet), set `variant="non_ema"` to download the `non_ema` file.

=== "fp16"

    ```py
    from mindone.diffusers import DiffusionPipeline
    import mindspore as ms

    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16", mindspore_dtype=ms.float16, use_safetensors=True
    )
    ```

=== "non-EMA"

    ```py
    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", variant="non_ema", use_safetensors=True
    )
    ```

Use the `variant` parameter in the [`save_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.save_pretrained) method to save a checkpoint as a different floating point type or as a non-EMA variant. You should try save a variant to the same folder as the original checkpoint, so you have the option of loading both from the same folder.

=== "fp16"

    ```python
    from mindone.diffusers import DiffusionPipeline

    pipeline.save_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16")
    ```

=== "non_ema"

    ```py
    pipeline.save_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="non_ema")
    ```

If you don't save the variant to an existing folder, you must specify the `variant` argument otherwise it'll throw an `Exception` because it can't find the original checkpoint.

```python
# 👎 this won't work
pipeline = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", mindspore_dtype=mindspore.float16, use_safetensors=True
)
# 👍 this works
pipeline = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", variant="fp16", mindspore_dtype=mindspore.float16, use_safetensors=True
)
```

## DiffusionPipeline explained

As a class method, [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) is responsible for two things:

- Download the latest version of the folder structure required for inference and cache it. If the latest folder structure is available in the local cache, [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) reuses the cache and won't redownload the files.
- Load the cached weights into the correct pipeline [class](../api/pipelines/overview.md#diffusers-summary) - retrieved from the `model_index.json` file - and return an instance of it.

The pipelines' underlying folder structure corresponds directly with their class instances. For example, the [`StableDiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/text2img/#mindone.diffusers.StableDiffusionPipeline) corresponds to the folder structure in [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

```python
from mindone.diffusers import DiffusionPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
print(pipeline)
```

You'll see pipeline is an instance of [`StableDiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/text2img/#mindone.diffusers.StableDiffusionPipeline), which consists of seven components:

- `"feature_extractor"`: a [`~transformers.CLIPImageProcessor`] from 🤗 Transformers.
- `"safety_checker"`: a [component](https://github.com/mindspore-lab/mindone/blob/master/mindone/diffusers/pipelines/stable_diffusion/safety_checker.py) for screening against harmful content.
- `"scheduler"`: an instance of [`PNDMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/pndm/#mindone.diffusers.PNDMScheduler).
- `"text_encoder"`: a [`~transformers.CLIPTextModel`] from 🤗 Transformers.
- `"tokenizer"`: a [`~transformers.CLIPTokenizer`] from 🤗 Transformers.
- `"unet"`: an instance of [`UNet2DConditionModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d-cond/#mindone.diffusers.UNet2DConditionModel).
- `"vae"`: an instance of [`AutoencoderKL`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/autoencoderkl/#mindone.diffusers.AutoencoderKL).

```json
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.29.2",
  "_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "image_encoder": [
    null,
    null
  ],
  "requires_safety_checker": true,
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
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

Compare the components of the pipeline instance to the [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) folder structure, and you'll see there is a separate folder for each of the components in the repository:

```
.
├── feature_extractor
│   └── preprocessor_config.json
├── model_index.json
├── safety_checker
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   ├── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── scheduler
│   └── scheduler_config.json
├── text_encoder
│   ├── config.json
|   ├── model.fp16.safetensors
│   ├── model.safetensors
│   |── pytorch_model.bin
|   └── pytorch_model.fp16.bin
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
|   |── diffusion_pytorch_model.fp16.bin
│   |── diffusion_pytorch_model.f16.safetensors
│   |── diffusion_pytorch_model.non_ema.bin
│   |── diffusion_pytorch_model.non_ema.safetensors
│   └── diffusion_pytorch_model.safetensors
|── vae
.   ├── config.json
.   ├── diffusion_pytorch_model.bin
    ├── diffusion_pytorch_model.fp16.bin
    ├── diffusion_pytorch_model.fp16.safetensors
    └── diffusion_pytorch_model.safetensors
```

You can access each of the components of the pipeline as an attribute to view its configuration:

```py
pipeline.tokenizer
CLIPTokenizer(
    name_or_path='/root/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/f03de327dd89b501a01da37fc5240cf4fdba85a1/tokenizer',
    vocab_size=49408,
    model_max_length=77,
    is_fast=False,
    padding_side='right',
    truncation_side='right',
    special_tokens={
        'bos_token': '<|startoftext|>',
        'eos_token': '<|endoftext|>',
        'unk_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>'},
    clean_up_tokenization_spaces=True
),
added_tokens_decoder={
    49406: AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
    49407: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
}
```

Every pipeline expects a [`model_index.json`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/model_index.json) file that tells the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline):

- which pipeline class to load from `_class_name`
- which version of 🧨 Diffusers was used to create the model in `_diffusers_version`
- what components from which library are stored in the subfolders (`name` corresponds to the component and subfolder name, `library` corresponds to the name of the library to load the class from, and `class` corresponds to the class name)

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
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
