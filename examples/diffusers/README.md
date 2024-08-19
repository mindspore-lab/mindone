
# Make 🤗 D🧨ffusers run on MindSpore

> State-of-the-art diffusion models for image and audio generation in MindSpore.
> We've tried to provide a completely consistent interface and usage with the [huggingface/diffusers](https://github.com/huggingface/diffusers).
> Only necessary changes are made to the [huggingface/diffusers](https://github.com/huggingface/diffusers) to make it seamless for users from torch.




## Quickstart

Generating outputs is super easy with 🤗 Diffusers. To generate an image from text, use the `from_pretrained` method to load any pretrained diffusion model (browse the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) for 19000+ checkpoints):

```diff
- from diffusers import DiffusionPipeline
+ from mindone.diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
-    torch_dtype=torch.float16,
+    mindspore_dtype=mindspore.float16
    use_safetensors=True
)

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt)[0][0]
```

official supported mindone.diffusers examples(follow hf diffusers):

 - [**Unconditional Image Generation**](./unconditional_image_generation)
 - [**Text-to-Image fine-tuning**](./text_to_image)
 - [**Textual Inversion**](./textual_inversion)
 - [**Dreambooth**](./dreambooth)
 - [**ControlNet**](./controlnet)
