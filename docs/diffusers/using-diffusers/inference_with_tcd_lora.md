<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Trajectory Consistency Distillation-LoRA

Trajectory Consistency Distillation (TCD) enables a model to generate higher quality and more detailed images with fewer steps. Moreover, owing to the effective error mitigation during the distillation process, TCD demonstrates superior performance even under conditions of large inference steps.

The major advantages of TCD are:

- Better than Teacher: TCD demonstrates superior generative quality at both small and large inference steps and exceeds the performance of [DPM-Solver++(2S)](../api/schedulers/multistep_dpm_solver.md) with Stable Diffusion XL (SDXL). There is no additional discriminator or LPIPS supervision included during TCD training.

- Flexible Inference Steps: The inference steps for TCD sampling can be freely adjusted without adversely affecting the image quality.

- Freely change detail level: During inference, the level of detail in the image can be adjusted with a single hyperparameter, *gamma*.

!!! tip

    For more technical details of TCD, please refer to the [paper](https://arxiv.org/abs/2402.19159) or official [project page](https://mhh0318.github.io/tcd/)).

For large models like SDXL, TCD is trained with [LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) to reduce memory usage. This is also useful because you can reuse LoRAs between different finetuned models, as long as they share the same base model, without further training.



This guide will show you how to perform inference with TCD-LoRAs for a variety of tasks like text-to-image and inpainting, as well as how you can easily combine TCD-LoRAs with other adapters. Choose one of the supported base model and it's corresponding TCD-LoRA checkpoint from the table below to get started.

| Base model                                                                                      | TCD-LoRA checkpoint                                            |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)                  | [TCD-SD15](https://huggingface.co/h1t/TCD-SD15-LoRA)           |
| [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)       | [TCD-SD21-base](https://huggingface.co/h1t/TCD-SD21-base-LoRA) |
| [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | [TCD-SDXL](https://huggingface.co/h1t/TCD-SDXL-LoRA)           |

## General tasks

In this guide, let's use the [`StableDiffusionXLPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl/#mindone.diffusers.StableDiffusionXLPipeline) and the [`TCDScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/tcd/#mindone.diffusers.TCDScheduler). Use the [`load_lora_weights`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/stable_diffusion/text2img/#mindone.diffusers.StableDiffusionPipeline) method to load the SDXL-compatible TCD-LoRA weights.

A few tips to keep in mind for TCD-LoRA inference are to:

- Keep the `num_inference_steps` between 4 and 50
- Set `eta` (used to control stochasticity at each step) between 0 and 1. You should use a higher `eta` when increasing the number of inference steps, but the downside is that a larger `eta` in [`TCDScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/tcd/#mindone.diffusers.TCDScheduler) leads to blurrier images. A value of 0.3 is recommended to produce good results.

=== "text-to-image"

    ```python
    import mindspore as ms
    from mindone.diffusers import StableDiffusionXLPipeline, TCDScheduler
    import numpy as np

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, mindspore_dtype=ms.float16)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()

    prompt = "Painting of the orange cat Otto von Garfield, Count of Bismarck-Schönhausen, Duke of Lauenburg, Minister-President of Prussia. Depicted wearing a Prussian Pickelhaube and eating his favorite meal - lasagna."

    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0,
        eta=0.3,
        generator=np.random.Generator(np.random.PCG64(0)),
    )[0][0]
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/511cc0e9-ddf3-490f-adaf-d4e7aacc3f6a"/>
    </div>

=== "inpainting"

    ```python
    import mindspore as ms
    from mindone.diffusers import StableDiffusionXLInpaintPipeline, TCDScheduler
    from mindone.diffusers.utils import load_image, make_image_grid
    import numpy as np

    base_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base_model_id, mindspore_dtype=ms.float16, variant="fp16")
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    init_image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    prompt = "a tiger sitting on a park bench"

    image = pipe(
      prompt=prompt,
      image=init_image,
      mask_image=mask_image,
      num_inference_steps=8,
      guidance_scale=0,
      eta=0.3,
      strength=0.99,  # make sure to use `strength` below 1.0
      generator=np.random.Generator(np.random.PCG64(0)),
    )[0][0]

    grid_image = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/c005fab2-ad5b-4753-9242-d99dbb306c81"/>
    </div>

## Community models

TCD-LoRA also works with many community finetuned models and plugins. For example, load the [animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0) checkpoint which is a community finetuned version of SDXL for generating anime images.

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionXLPipeline, TCDScheduler
import numpy as np

base_model_id = "cagliostrolab/animagine-xl-3.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, mindspore_dtype=ms.float16, variant="fp16")
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "A man, clad in a meticulously tailored military uniform, stands with unwavering resolve. The uniform boasts intricate details, and his eyes gleam with determination. Strands of vibrant, windswept hair peek out from beneath the brim of his cap."

image = pipe(
    prompt=prompt,
    num_inference_steps=8,
    guidance_scale=0,
    eta=0.3,
    generator=np.random.Generator(np.random.PCG64(0)),
)[0][0]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/395ee969-f129-4cbd-a3fb-0439f0d56119"/>
</div>

TCD-LoRA also supports other LoRAs trained on different styles. For example, let's load the [TheLastBen/Papercut_SDXL](https://huggingface.co/TheLastBen/Papercut_SDXL) LoRA and fuse it with the TCD-LoRA with the [`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters) method.

!!! tip

    Check out the [Merge LoRAs](merge_loras.md) guide to learn more about efficient merging methods.

```python
import mindspore as ms
from mindone.diffusers import StableDiffusionXLPipeline, TCDScheduler
import numpy as np

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"
styled_lora_id = "TheLastBen/Papercut_SDXL"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, mindspore_dtype=ms.float16)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id, adapter_name="tcd")
pipe.load_lora_weights(styled_lora_id, adapter_name="style")
pipe.set_adapters(["tcd", "style"], adapter_weights=[1.0, 1.0])

prompt = "papercut of a winter mountain, snow"

image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    generator=np.random.Generator(np.random.PCG64(0)),
)[0][0]
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/4753329b-d626-4b1e-a001-bb4a2a94429d"/>
</div>

## Adapters

TCD-LoRA is very versatile, and it can be combined with other adapter types like ControlNets, IP-Adapter, and AnimateDiff.

=== "ControlNet"

    ### Depth ControlNet

    ```python
    import mindspore as ms
    from mindspore import ops
    import numpy as np
    from PIL import Image
    from transformers import DPTImageProcessor
    from mindone.transformers import DPTForDepthEstimation
    from mindone.diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, TCDScheduler
    from mindone.diffusers.utils import load_image, make_image_grid

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

    def get_depth_map(image):
        image = feature_extractor(images=image, return_tensors="np").pixel_values
        image = ms.Tensor(image)

        depth_map = depth_estimator(image)[0]

        depth_map = ops.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = ops.amin(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_max = ops.amax(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = ops.cat([depth_map] * 3, axis=1)

        image = image.permute(0, 2, 3, 1).asnumpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_id = "diffusers/controlnet-depth-sdxl-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        mindspore_dtype=ms.float16,
        variant="fp16",
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        mindspore_dtype=ms.float16,
    )

    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()

    prompt = "stormtrooper lecture, photorealistic"

    image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
    depth_image = get_depth_map(image)

    controlnet_conditioning_scale = 0.5  # recommended for good generalization

    image = pipe(
        prompt,
        image=depth_image,
        num_inference_steps=4,
        guidance_scale=0,
        eta=0.3,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=np.random.Generator(np.random.PCG64(42)),
    )[0][0]

    grid_image = make_image_grid([depth_image, image], rows=1, cols=2)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/e456fef0-05d7-453c-8071-61d35d12cbec"/>
    </div>

    ### Canny ControlNet
    ```python
    import mindspore as ms
    from mindone.diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, TCDScheduler
    from mindone.diffusers.utils import load_image, make_image_grid
    import numpy as np

    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        mindspore_dtype=ms.float16,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        mindspore_dtype=ms.float16,
    )

    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()

    prompt = "ultrarealistic shot of a furry blue bird"

    canny_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png")

    controlnet_conditioning_scale = 0.5  # recommended for good generalization

    image = pipe(
        prompt,
        image=canny_image,
        num_inference_steps=4,
        guidance_scale=0,
        eta=0.3,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=np.random.Generator(np.random.PCG64(0)),
    )[0][0]

    grid_image = make_image_grid([canny_image, image], rows=1, cols=2)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/b6433aff-45a9-4f1c-abca-3734181876ba"/>
    </div>

    !!! tip

        The inference parameters in this example might not work for all examples, so we recommend you to try different values for `num_inference_steps`, `guidance_scale`, `controlnet_conditioning_scale` and `cross_attention_kwargs` parameters and choose the best one.

=== "IP-Adapter"

    This example shows how to use the TCD-LoRA with SDXL.

    ```python
    import mindspore as ms
    from mindone.diffusers import StableDiffusionXLPipeline, TCDScheduler
    from mindone.diffusers.utils import load_image, make_image_grid
    import numpy as np

    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    tcd_lora_id = "h1t/TCD-SDXL-LoRA"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        mindspore_dtype=ms.float16,
    )
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    pipe.load_lora_weights(tcd_lora_id)
    pipe.fuse_lora()

    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors")
    pipe.set_ip_adapter_scale(0.5)

    ref_image = load_image("https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png").resize((512, 512))

    prompt = "best quality, high quality, wearing sunglasses"

    generator = np.random.Generator(np.random.PCG64(0))

    image = pipe(
        prompt=prompt,
        ip_adapter_image=ref_image,
        guidance_scale=0,
        num_inference_steps=50,
        generator=generator,
        eta=0.3,
    )[0][0]

    grid_image = make_image_grid([ref_image, image], rows=1, cols=2)
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/ad81a47f-4d2b-4c2f-8392-86ed4d7a3a96"/>
    </div>

=== "AnimateDiff"

    [`AnimateDiff`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/animatediff/#mindone.diffusers.AnimateDiffPipeline) allows animating images using Stable Diffusion models. TCD-LoRA can substantially accelerate the process without degrading image quality. The quality of animation with TCD-LoRA and AnimateDiff has a more lucid outcome.

    ```python
    import mindspore as ms
    from mindone.diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler, TCDScheduler
    from mindone.diffusers.utils import export_to_gif
    import numpy as np

    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")
    pipe = AnimateDiffPipeline.from_pretrained(
        "frankjoshua/toonyou_beta6",
        motion_adapter=adapter,
    )

    # set TCDScheduler
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

    # load TCD LoRA
    pipe.load_lora_weights("h1t/TCD-SD15-LoRA", adapter_name="tcd")
    pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", weight_name="diffusion_pytorch_model.safetensors", adapter_name="motion-lora")

    pipe.set_adapters(["tcd", "motion-lora"], adapter_weights=[1.0, 1.2])

    prompt = "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress"
    generator = np.random.Generator(np.random.PCG64(0))
    frames = pipe(
        prompt=prompt,
        num_inference_steps=5,
        guidance_scale=0,
        num_frames=24,
        eta=0.3,
        generator=generator
    )[0][0]
    export_to_gif(frames, "animation.gif")
    ```

    <div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
        <img class="rounded-xl" src="https://github.com/user-attachments/assets/865cdfac-0579-4c82-a0b0-88c9047b08fa"/>
    </div>
