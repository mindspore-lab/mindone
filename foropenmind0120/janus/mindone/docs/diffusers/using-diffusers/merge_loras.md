<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Merge LoRAs

It can be fun and creative to use multiple [LoRAs]((https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora)) together to generate something entirely new and unique. This works by merging multiple LoRA weights together to produce images that are a blend of different styles. Diffusers provides a few methods to merge LoRAs depending on *how* you want to merge their weights, which can affect image quality.

This guide will show you how to merge LoRAs using the [`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters) and [`~peft.LoraModel.add_weighted_adapter`] methods. To improve inference speed and reduce memory-usage of merged LoRAs, you'll also see how to use the [`fuse_lora`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.fuse_lora) method to fuse the LoRA weights with the original weights of the underlying model.

For this guide, load a Stable Diffusion XL (SDXL) checkpoint and the [KappaNeuro/studio-ghibli-style](https://huggingface.co/KappaNeuro/studio-ghibli-style) and [Norod78/sdxl-chalkboarddrawing-lora](https://huggingface.co/Norod78/sdxl-chalkboarddrawing-lora) LoRAs with the [`load_lora_weights`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.load_lora_weights) method. You'll need to assign each LoRA an `adapter_name` to combine them later.

```py
from mindone.diffusers import DiffusionPipeline
import mindspore as ms
import numpy as np

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16)
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")
```

## set_adapters

The [`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters) method merges LoRA adapters by concatenating their weighted matrices. Use the adapter name to specify which LoRAs to merge, and the `adapter_weights` parameter to control the scaling for each LoRA. For example, if `adapter_weights=[0.5, 0.5]`, then the merged LoRA output is an average of both LoRAs. Try adjusting the adapter weights to see how it affects the generated image!

```py
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])

generator = np.random.Generator(np.random.PCG64(0))
prompt = "A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai"
image = pipeline(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0})[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/d947dae6-e825-4478-89e6-02cdacf92c46"/>
</div>

## fuse_lora

[`set_adapters`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/unet/#mindone.diffusers.loaders.unet.UNet2DConditionLoadersMixin.set_adapters) method require loading the base model and the LoRA adapters separately which incurs some overhead. The [`fuse_lora`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.fuse_lora) method allows you to fuse the LoRA weights directly with the original weights of the underlying model. This way, you're only loading the model once which can increase inference and lower memory-usage.

You can use PEFT to easily fuse/unfuse multiple adapters directly into the model weights (both UNet and text encoder) using the [`fuse_lora`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.fuse_lora) method, which can lead to a speed-up in inference and lower VRAM usage.

For example, if you have a base model and adapters loaded and set as active with the following adapter weights:

```py
from mindone.diffusers import DiffusionPipeline
import mindspore as ms

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16)
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")

pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
```

Fuse these LoRAs into the UNet with the [`fuse_lora`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.fuse_lora) method. The `lora_scale` parameter controls how much to scale the output by with the LoRA weights. It is important to make the `lora_scale` adjustments in the [`fuse_lora`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.fuse_lora) method because it won’t work if you try to pass `scale` to the `cross_attention_kwargs` in the pipeline.

```py
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
```

Then you should use [`unload_lora_weights`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.unload_lora_weights) to unload the LoRA weights since they've already been fused with the underlying base model. Finally, call [`save_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.save_pretrained) to save the fused pipeline locally or you could call [`push_to_hub`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.push_to_hub) to push the fused pipeline to the Hub.

```py
pipeline.unload_lora_weights()
# save locally
pipeline.save_pretrained("path/to/fused-pipeline")
# save to the Hub
pipeline.push_to_hub("fused-ikea-feng")
```

Now you can quickly load the fused pipeline and use it for inference without needing to separately load the LoRA adapters.

```py
pipeline = DiffusionPipeline.from_pretrained(
    "username/fused-ikea-feng", mindspore_dtype=ms.float16,
)

image = pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", generator=np.random.Generator(np.random.PCG64(0)))[0][0]
image
```

You can call [`unfuse_lora`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/loaders/lora/#mindone.diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.unfuse_lora) to restore the original model's weights (for example, if you want to use a different `lora_scale` value). However, this only works if you've only fused one LoRA adapter to the original model. If you've fused multiple LoRAs, you'll need to reload the model.

```py
pipeline.unfuse_lora()
```
