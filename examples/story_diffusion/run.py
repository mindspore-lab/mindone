import copy
import json
import os
import sys

import numpy as np

import mindspore as ms

style_list = json.load(open("utils/style_template.json", "r"))
styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../..")
sys.path.insert(0, mindone_lib_path)

import argparse

from utils.gradio_utils import SpatialAttnProcessor2_0, get_attention_mask_dict

from mindone.diffusers import StableDiffusionXLPipeline
from mindone.diffusers.models.attention_processor import AttnProcessor
from mindone.diffusers.schedulers import DDIMScheduler
from mindone.utils.seed import set_random_seed

STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"  # "Japanese Anime"

models_dict = {
    "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
    "RealVision": "SG161222/RealVisXL_V4.0",
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y",
}


def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + " " + negative


def init_env(mode, device_target):
    # no parallel mode currently
    ms.set_context(mode=mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=mode,
        device_target=device_target,
        device_id=device_id,
    )

    return device_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--height",
        default=768,
        type=int,
        help="The generated image height",
    )
    parser.add_argument(
        "--width",
        default=768,
        type=int,
        help="The generated image width",
    )
    parser.add_argument(
        "--mode",
        default=1,
        type=int,
        help="The mindspore mode: 1 means pynative and 0 means graph mode. Defaults to pynative mode.",
    )
    parser.add_argument("--sd_model_name", type=str, default="RealVision", choices=list(models_dict.keys()))
    parser.add_argument("--style_name", type=str, default="Comic book", choices=list(models_dict.keys()))
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache_dir",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The directory where the generated images will be stored in.",
    )
    parser.add_argument(
        "--general_prompt", type=str, default="a man with a black suit", help="The prompt of the person/object"
    )
    parser.add_argument("--sampling_steps", type=int, default=25, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="the scale for classifier-free guidance")
    parser.add_argument("--seed", default=3407, type=int, help="random seed")
    args = parser.parse_args()
    return args


def insert_paired_attention(unet, id_length, atten_mask_dict):
    # Insert PairedAttention
    count_attn = 0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        # if name.startswith("mid_block"):
        #     hidden_size = unet.config.block_out_channels[-1]
        # elif name.startswith("up_blocks"):
        #     block_id = int(name[len("up_blocks.")])
        #     hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        # elif name.startswith("down_blocks"):
        #     block_id = int(name[len("down_blocks.")])
        #     hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None and (name.startswith("up_blocks")):
            attn_procs[name] = SpatialAttnProcessor2_0(
                id_length=id_length,
                attention_masks=atten_mask_dict,
            )
            count_attn += 1
        else:
            attn_procs[name] = AttnProcessor()
    print("successsfully load consistent self-attention")
    print(f"number of replaced processors: {count_attn}")
    unet.set_attn_processor(copy.deepcopy(attn_procs))
    return unet


def set_unet_variable(unet, target_name, target_value):
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if cross_attention_dim is None and (name.startswith("up_blocks")):
            assert hasattr(unet.attn_processors[name], target_name)
            setattr(unet.attn_processors[name], target_name, target_value)
    print(f"Set `{target_name}` for all consistent self-attentions to be {target_value}")


def load_sdxl_pipeline(args):
    sd_model_path = models_dict[args.sd_model_name]  # "SG161222/RealVisXL_V4.0"
    # LOAD Stable Diffusion Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sd_model_path,
        cache_dir=args.cache_dir,
        mindspore_dtype=ms.float16,
        use_safetensors=False,
        local_files_only=True,
    )
    # pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.sampling_steps)
    return pipe


def parse_prompts(args):
    # parse prompts
    negative_prompt = (
        "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb,"
        + " floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
    )
    prompt_array = [
        "wake up in the bed",
        "have breakfast",
        "is on the road, go to the company",
        "work in the company",
        "running in the playground",
        "reading book in the home",
    ]

    prompts = [args.general_prompt + "," + prompt for prompt in prompt_array]
    id_prompts = prompts[:id_length]
    real_prompts = prompts[id_length:]
    id_prompts, negative_prompt = apply_style(args.style_name, id_prompts, negative_prompt)
    return id_prompts, real_prompts, negative_prompt


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    id_length = 4
    total_length = id_length + 1

    pipe = load_sdxl_pipeline(args)
    unet = pipe.unet
    # test pipeline if needed
    # prompt = "An astronaut riding a green horse"
    # images = pipe(prompt=prompt)[0]
    # images[0].save("image.png")

    # strength of consistent self-attention: the larger, the stronger
    threshold_32 = 0.5
    threshold_64 = 0.5
    atten_mask_dict = get_attention_mask_dict(
        down_scales_list=[32, 16],
        thresholds_list=[threshold_32, threshold_64],
        total_length=total_length,
        id_length=id_length,
        height=args.height,
        width=args.width,
        dtype=ms.float16,
    )
    unet = insert_paired_attention(unet, id_length, atten_mask_dict)

    id_prompts, real_prompts, negative_prompt = parse_prompts(args)
    # write = True, memorizing
    set_unet_variable(unet, "write", True)
    set_unet_variable(unet, "cur_step", 0)
    generator = np.random.Generator(np.random.PCG64(args.seed))
    id_images = pipe(
        id_prompts,
        num_inference_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        negative_prompt=negative_prompt,
        generator=generator,
    )[0]
    set_unet_variable(unet, "cur_step", 1)  # update cur_step by 1 after one inference

    # write = False
    set_unet_variable(unet, "write", False)
    for i, id_image in enumerate(id_images):
        save_fp = os.path.join(args.output_dir, f"id_{i}-{id_prompts[i][:100]}.png")
        id_image.save(save_fp)

    real_images = []
    for real_prompt in real_prompts:
        set_unet_variable(unet, "cur_step", 0)
        real_prompt = apply_style_positive(args.style_name, real_prompt)
        real_images.extend(
            pipe(
                real_prompt,
                num_inference_steps=args.sampling_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                negative_prompt=negative_prompt,
                generator=generator,
            )[0]
        )

    for i, real_image in enumerate(real_images):
        # display(real_image)
        save_fp = os.path.join(args.output_dir, f"real_{i}-{real_prompts[i][:100]}.png")
        id_image.save(save_fp)

    new_prompt_array = ["siting on the sofa", "on the bed, at night "]
    new_prompts = [args.general_prompt + "," + prompt for prompt in new_prompt_array]
    new_images = []
    for new_prompt in new_prompts:
        set_unet_variable(unet, "cur_step", 0)
        new_prompt = apply_style_positive(args.style_name, new_prompt)
        new_images.extend(
            pipe(
                new_prompt,
                num_inference_steps=args.sampling_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                negative_prompt=negative_prompt,
                generator=generator,
            )[0]
        )
        for i, real_image in enumerate(new_images):
            # display(real_image)
            save_fp = os.path.join(args.output_dir, f"new_{i}-{new_prompts[i][:100]}.png")
            id_image.save(save_fp)
