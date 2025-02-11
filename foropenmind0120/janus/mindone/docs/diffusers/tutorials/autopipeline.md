<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

Diffusers provides many pipelines for basic tasks like generating images, videos, audio, and inpainting. On top of these, there are specialized pipelines for adapters and features like upscaling, super-resolution, and more. Different pipeline classes can even use the same checkpoint because they share the same pretrained model! With so many different pipelines, it can be overwhelming to know which pipeline class to use.

The [AutoPipeline](../api/pipelines/auto_pipeline.md) class is designed to simplify the variety of pipelines in Diffusers. It is a generic *task-first* pipeline that lets you focus on a task ([`AutoPipelineForText2Image`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/auto_pipeline/#mindone.diffusers.AutoPipelineForText2Image), [`AutoPipelineForImage2Image`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/auto_pipeline/#mindone.diffusers.AutoPipelineForImage2Image), and [`AutoPipelineForInpainting`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/auto_pipeline/#mindone.diffusers.AutoPipelineForInpainting)) without needing to know the specific pipeline class. The [AutoPipeline](../api/pipelines/auto_pipeline.md) automatically detects the correct pipeline class to use.

For example, let's use the [dreamlike-art/dreamlike-photoreal-2.0](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0) checkpoint.

Under the hood, [AutoPipeline](../api/pipelines/auto_pipeline.md):

1. Detects a `"stable-diffusion"` class from the [model_index.json](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0/blob/main/model_index.json) file.
2. Depending on the task you're interested in, it loads the [`StableDiffusionPipeline`](pipelines/stable_diffusion/text2img.md#mindone.diffusers.StableDiffusionPipeline), [`StableDiffusionImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/img2img/#mindone.diffusers.StableDiffusionImg2ImgPipeline), or [`StableDiffusionInpaintPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/inpaint/#mindone.diffusers.StableDiffusionInpaintPipeline). Any parameter (`strength`, `num_inference_steps`, etc.) you would pass to these specific pipelines can also be passed to the [AutoPipeline](../api/pipelines/auto_pipeline).

=== "text-to-image"

    ```py
    from mindone.diffusers import AutoPipelineForText2Image
    import mindspore as ms
    import numpy as np

    pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0", mindspore_dtype=ms.float16, use_safetensors=True
    )

    prompt = "cinematic photo of Godzilla eating sushi with a cat in a izakaya, 35mm photograph, film, professional, 4k, highly detailed"
    generator = np.random.Generator(np.random.PCG64(22))
    image = pipe_txt2img(prompt, generator=generator)[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img src="https://github.com/user-attachments/assets/efaa9a06-59b7-49a0-bdc6-5d477f351310"/>
    </div>


=== "image-to-image"

    ```py
    from mindone.diffusers import AutoPipelineForImage2Image
    from mindone.diffusers.utils import load_image
    import mindspore as ms
    import numpy as np

    pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
        "dreamlike-art/dreamlike-photoreal-2.0", mindspore_dtype=ms.float16, use_safetensors=True
    )

    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png")

    prompt = "cinematic photo of Godzilla eating burgers with a cat in a fast food restaurant, 35mm photograph, film, professional, 4k, highly detailed"
    generator = np.random.Generator(np.random.PCG64(7))
    image = pipe_img2img(prompt, image=init_image, generator=generator)[0][0]
    image
    ```

    Notice how the [dreamlike-art/dreamlike-photoreal-2.0](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0) checkpoint is used for both text-to-image and image-to-image tasks? To save memory and avoid loading the checkpoint twice, use the [`from_pipe`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pipe) method.

    ```py
    pipe_img2img = AutoPipelineForImage2Image.from_pipe(pipe_txt2img)
    image = pipeline(prompt, image=init_image, generator=generator)[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img src="https://github.com/user-attachments/assets/de3baa9e-b95f-462a-b2a0-5dc2f63cabb5"/>
    </div>

=== "inpainting"

    ```py
    from mindone.diffusers import AutoPipelineForInpainting
    from mindone.diffusers.utils import load_image
    import mindspore as ms
    import numpy as np

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16, use_safetensors=True
    )

    init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png")
    mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-mask.png")

    prompt = "cinematic photo of a owl, 35mm photograph, film, professional, 4k, highly detailed"
    generator = np.random.Generator(np.random.PCG64(38))
    image = pipeline(prompt, image=init_image, mask_image=mask_image, generator=generator, strength=0.4)[0][0]
    image
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img src="https://github.com/user-attachments/assets/8d2dc543-4702-4979-a4df-7f66f4d3d00d"/>
    </div>

## Unsupported checkpoints

The [AutoPipeline](../api/pipelines/auto_pipeline.md) supports [Stable Diffusion](../api/pipelines/stable_diffusion/overview.md), [Stable Diffusion XL](../api/pipelines/stable_diffusion/stable_diffusion_xl.md), [ControlNet](../api/pipelines/controlnet.md), [Kandinsky 2.1](../api/pipelines/kandinsky.md), [Kandinsky 2.2](../api/pipelines/kandinsky_v22.md), and [DeepFloyd IF](../api/pipelines/deepfloyd_if.md) checkpoints.

If you try to load an unsupported checkpoint, you'll get an error.

```py
from mindone.diffusers import AutoPipelineForImage2Image
import mindspore as ms

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", mindspore_dtype=ms.float16, use_safetensors=True
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```
