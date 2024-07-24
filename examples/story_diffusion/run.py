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
from utils.gradio_utils import SpatialAttnProcessor2_0, cal_attn_mask_xl

from mindone.diffusers import StableDiffusionXLPipeline
from mindone.diffusers.models.attention_processor import AttnProcessor
from mindone.diffusers.schedulers import DDIMScheduler
from mindone.utils.seed import set_random_seed


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


init_env(mode=1, device_target="GPU")
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
MAX_SEED = np.iinfo(np.int32).max
global models_dict
use_va = False
models_dict = {
    "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
    "RealVision": "SG161222/RealVisXL_V4.0",
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y",
}

global attn_count, total_count, id_length, total_length, cur_step, cur_model_type
global write
global sa32, sa64
global height, width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
global attn_procs, unet
attn_procs = {}
#
write = False
# strength of consistent self-attention: the larger, the stronger
sa32 = 0.5
sa64 = 0.5
# Res. of the Generated Comics. Please Note: SDXL models may do worse in a low-resolution!
height = 768
width = 768
#
global pipe
global sd_model_path
sd_model_path = models_dict["RealVision"]  # "SG161222/RealVisXL_V4.0"
cache_dir = "./cache_dir"

# LOAD Stable Diffusion Pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    sd_model_path, cache_dir=cache_dir, mindspore_dtype=ms.float16, use_safetensors=False, local_files_only=True
)
# pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
unet = pipe.unet
prompt = "An astronaut riding a green horse"
# breakpoint()
images = pipe(prompt=prompt)
# Insert PairedAttention
for name in unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None and (name.startswith("up_blocks")):
        attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
        total_count += 1
    else:
        attn_procs[name] = AttnProcessor()
print("successsfully load consistent self-attention")
print(f"number of replaced processors: {total_count}")
unet.set_attn_processor(copy.deepcopy(attn_procs))
global mask1024, mask4096
mask1024, mask4096 = cal_attn_mask_xl(total_length, id_length, sa32, sa64, height, width, dtype=ms.float16)

guidance_scale = 5.0
seed = 2047
id_length = 4
num_steps = 50
general_prompt = "a man with a black suit"
negative_prompt = (
    "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly"
    + ", disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
)
prompt_array = [
    "wake up in the bed",
    "have breakfast",
    "is on the road, go to the company",
    "work in the company",
    "running in the playground",
    "reading book in the home",
]


def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + " " + negative


# Set the generated Style
style_name = "Comic book"
set_random_seed(seed)
generator = generator = np.random.Generator(np.random.PCG64(seed))
prompts = [general_prompt + "," + prompt for prompt in prompt_array]
id_prompts = prompts[:id_length]
real_prompts = prompts[id_length:]

write = True
cur_step = 0
attn_count = 0
id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
id_images = pipe(
    id_prompts,
    num_inference_steps=num_steps,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    negative_prompt=negative_prompt,
    generator=generator,
)[0]

write = False
for id_image in id_images:
    pass
real_images = []
for real_prompt in real_prompts:
    cur_step = 0
    real_prompt = apply_style_positive(style_name, real_prompt)
    real_images.append(
        pipe(
            real_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]
    )
for real_image in real_images:
    # display(real_image)
    pass
