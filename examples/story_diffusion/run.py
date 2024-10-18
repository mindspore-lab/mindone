import copy
import json
import os
import random
import sys
from typing import Optional

import numpy as np

import mindspore as ms
from mindspore import ops

style_list = json.load(open("utils/style_template.json", "r"))
styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../..")
sys.path.insert(0, mindone_lib_path)

import argparse

from PIL import ImageFont
from utils.gradio_utils import AttnProcessor2_0 as AttnProcessor
from utils.gradio_utils import cal_attn_mask_xl
from utils.utils import get_comic_4panel

from mindone.diffusers import StableDiffusionXLPipeline
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


class SpatialAttnProcessor2_0:
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        dtype=ms.float16,
    ):
        super().__init__()

        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def scaled_dot_product_attention(
        self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, training=False
    ) -> ms.Tensor:
        L, S = query.shape[-2], key.shape[-2]
        scale_factor = 1 / (query.shape[-1] ** 0.5) if scale is None else scale
        _dtype = query.dtype
        attn_bias = ops.zeros((L, S), dtype=ms.float32)
        if is_causal:
            assert attn_mask is None
            temp_mask = ops.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias = attn_bias.masked_fill(~temp_mask, -1e5)
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                attn_bias = attn_bias.masked_fill(~attn_mask, -1e5)
            else:
                attn_bias += attn_mask
        attn_weight = ops.matmul(query, key.swapaxes(-2, -1)) * scale_factor
        attn_weight = attn_weight.to(ms.float32)
        attn_weight += attn_bias
        attn_weight = ops.softmax(attn_weight, axis=-1)
        attn_weight = ops.dropout(attn_weight, p=dropout_p, training=training)
        out = ops.matmul(attn_weight.to(_dtype), value)
        out = out.astype(_dtype)
        return out

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        global write, cur_step, total_count, attn_count, mask1024, mask4096
        global sa32, sa64
        global height, width
        if write:
            self.id_bank[cur_step] = [hidden_states[: self.id_length], hidden_states[self.id_length :]]
        else:
            encoder_hidden_states = ops.cat(
                [self.id_bank[cur_step][0], hidden_states[:1], self.id_bank[cur_step][1], hidden_states[1:]]
            )
        # skip in early step
        if cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        else:  # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length :]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length :]
                else:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[
                            : mask1024.shape[0] // self.total_length * self.id_length,
                            : mask1024.shape[0] // self.total_length * self.id_length,
                        ]
                    else:
                        attention_mask = mask4096[
                            : mask4096.shape[0] // self.total_length * self.id_length,
                            : mask4096.shape[0] // self.total_length * self.id_length,
                        ]
                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(
                self.total_length, self.id_length, sa32, sa64, height, width, dtype=self.dtype
            )
        return hidden_states

    def __call1__(
        self,
        attn,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(
            -1, img_nums * nums_token, channel
        )

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, nums_token, channel).reshape(
                -1, (self.id_length + 1) * nums_token, channel
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        hidden_states = self.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def __call2__(
        self,
        attn,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length, channel
            ).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        hidden_states = self.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


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
    parser.add_argument("--style_name", type=str, default="Comic book", choices=STYLE_NAMES)
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
    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="the scale for classifier-free guidance")
    parser.add_argument("--seed", default=3407, type=int, help="random seed")
    parser.add_argument("--local_files_only", action="store_true", help="load from local Huggingface files.")
    args = parser.parse_args()
    return args


def load_sdxl_pipeline(args):
    sd_model_path = models_dict[args.sd_model_name]  # "SG161222/RealVisXL_V4.0"
    # LOAD Stable Diffusion Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sd_model_path,
        cache_dir=args.cache_dir,
        mindspore_dtype=ms.float16,
        use_safetensors=False,
        local_files_only=args.local_files_only,
    )
    # pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(args.sampling_steps)
    return pipe


if __name__ == "__main__":
    args = parse_args()
    init_env(args.mode, device_target="Ascend")
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    global attn_count, total_count, cur_step
    global write
    global height, width
    height = args.height
    width = args.width
    write = False
    total_count = 0
    attn_count = 0
    cur_step = 0
    id_length = 4
    total_length = id_length + 1

    global attn_procs, unet
    attn_procs = {}
    pipe = load_sdxl_pipeline(args)
    unet = pipe.unet
    # test pipeline if needed
    # prompt = "An astronaut riding a green horse"
    # images = pipe(prompt=prompt)[0]
    # images[0].save("image.png")

    # strength of consistent self-attention: the larger, the stronger
    sa32 = 0.5
    sa64 = 0.5
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
    print(f"number of the processor : {total_count}")
    unet.set_attn_processor(copy.deepcopy(attn_procs))

    global mask1024, mask4096
    mask1024, mask4096 = cal_attn_mask_xl(total_length, id_length, sa32, sa64, height, width, dtype=ms.float16)
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
    # write = True, memorizing
    write = True
    cur_step = 0
    attn_count = 0
    generator = np.random.Generator(np.random.PCG64(args.seed))
    id_images = pipe(
        id_prompts,
        num_inference_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        generator=generator,
    )[0]

    # write = False
    write = False
    for i, id_image in enumerate(id_images):
        save_fp = os.path.join(args.output_dir, f"id_{i}-{id_prompts[i][:100]}.png")
        id_image.save(save_fp)

    real_images = []
    for real_prompt in real_prompts:
        cur_step = 0
        real_prompt = apply_style_positive(args.style_name, real_prompt)
        real_images.extend(
            pipe(
                real_prompt,
                num_inference_steps=args.sampling_steps,
                guidance_scale=args.guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            )[0]
        )

    for i, real_image in enumerate(real_images):
        # display(real_image)
        save_fp = os.path.join(args.output_dir, f"real_{i}-{real_prompts[i][:100]}.png")
        real_image.save(save_fp)

    new_prompt_array = ["siting on the sofa", "on the bed, at night "]
    new_prompts = [args.general_prompt + "," + prompt for prompt in new_prompt_array]
    new_images = []
    for new_prompt in new_prompts:
        cur_step = 0
        new_prompt = apply_style_positive(args.style_name, new_prompt)
        new_images.extend(
            pipe(
                new_prompt,
                num_inference_steps=args.sampling_steps,
                guidance_scale=args.guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            )[0]
        )
    for i, new_image in enumerate(new_images):
        # display(real_image)
        save_fp = os.path.join(args.output_dir, f"new_{i}-{new_prompts[i][:100]}.png")
        new_image.save(save_fp)

    total_images = id_images + real_images + new_images
    # LOAD Fonts, can also replace with any Fonts you have!
    font = ImageFont.truetype("./fonts/Inkfree.ttf", 30)
    comics = get_comic_4panel(total_images, captions=prompts + new_prompts, font=font)

    for i, comic in enumerate(comics):
        comic.save(os.path.join(args.output_dir, f"{i}-{args.style_name}-{args.sd_model_name}.png"))
