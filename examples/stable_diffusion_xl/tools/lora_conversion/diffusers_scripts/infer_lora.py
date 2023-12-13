import os

import torch
from diffusers import AutoPipelineForText2Image
from fire import Fire

# import pdb


def infer(
    model_path="sd-cnart-model-lora-sdxl",
    prompt="a black and white photo of a butterfly",
    seed=43,
    images_per_prompt=1,
    num_inference_steps=40,
    guidance_scale=5.0,
):
    prompts = []
    if os.path.exists(prompt):
        with open(prompt, "r") as f:
            prompts = f.read().splitlines()
    else:
        prompts = [prompt]

    # model_path = "sd-pokemon-model-lora-sdxl"
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "models/sdxl_base", torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    if model_path.endswith(".ckpt"):
        pipeline_text2image.unet.load_attn_procs(model_path, use_safetensors=False)
    else:
        pipeline_text2image.unet.load_attn_procs(model_path)

    # torch.manual_seed(seed)
    output_dir = model_path + "-gen-images"
    os.makedirs(output_dir, exist_ok=True)

    for pr_i, prompt in enumerate(prompts):
        print(f"{pr_i+1}/{len(prompts)} prompt: ", prompt)

        generator = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(images_per_prompt)]
        # pdb.set_trace()
        images = pipeline_text2image(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024,
            num_images_per_prompt=images_per_prompt,
            guidance_scale=guidance_scale,
        ).images

        for i in range(images_per_prompt):
            images[i].save("{}/p{:03d}_img{:03d}.png".format(output_dir, pr_i, i))

    print("Generated images are saved in ", output_dir)


if __name__ == "__main__":
    Fire(infer)
