#!/usr/bin/env python
import argparse
import ast
import os
import sys
import time

from PIL import Image, ImageOps

sys.path.append("../stable_diffusion_xl/")
sys.path.append("../stable_diffusion_v2/")
import numpy as np
from gm.helpers import SD_XL_BASE_RATIOS, VERSION2SPECS, create_model, init_sampling, perform_save_locally
from gm.util import seed_everything
from omegaconf import OmegaConf
from transformers import CLIPImageProcessor

import mindspore as ms


def get_parser_sample():
    parser = argparse.ArgumentParser(description="sampling with sd-xl")
    parser.add_argument("--config", type=str, default="configs/inference/sd_xl_base.yaml")
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms_ip_adapter.ckpt")
    parser.add_argument("--ip_scale", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="best quality, high quality")
    parser.add_argument(
        "--negative_prompt", type=str, default="monochrome, lowres, bad anatomy, worst quality, low quality"
    )
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--orig_width", type=int, default=None)
    parser.add_argument("--orig_height", type=int, default=None)
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--crop_coords_top", type=int, default=None)
    parser.add_argument("--crop_coords_left", type=int, default=None)
    parser.add_argument("--aesthetic_score", type=float, default=None)
    parser.add_argument("--negative_aesthetic_score", type=float, default=None)
    parser.add_argument("--sampler", type=str, default="EulerEDMSampler")
    parser.add_argument("--guider", type=str, default="VanillaCFG")
    parser.add_argument("--discretization", type=str, default="DiffusersDDPMDiscretization")
    parser.add_argument("--sample_step", type=int, default=30)
    parser.add_argument("--num_cols", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="outputs/demo/", help="save dir")

    # system
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_jit", type=ast.literal_eval, default=True, help="use jit or not")
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    return parser


def load_clip_image(image: str) -> np.ndarray:
    image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = CLIPImageProcessor()(image)
    return image.pixel_values


def run_text2img(
    args, model, version_dict, is_legacy=False, return_latents=False, filter=None, stage2strength=None, amp_level="O0"
):
    assert args.sd_xl_base_ratios in SD_XL_BASE_RATIOS
    W, H = SD_XL_BASE_RATIOS[args.sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]

    clip_img = load_clip_image(args.img)

    value_dict = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "clip_img": clip_img,
        "orig_width": args.orig_width if args.orig_width else W,
        "orig_height": args.orig_height if args.orig_height else H,
        "target_width": args.target_width if args.target_width else W,
        "target_height": args.target_height if args.target_height else H,
        "crop_coords_top": max(args.crop_coords_top if args.crop_coords_top else 0, 0),
        "crop_coords_left": max(args.crop_coords_left if args.crop_coords_left else 0, 0),
        "aesthetic_score": args.aesthetic_score if args.aesthetic_score else 6.0,
        "negative_aesthetic_score": args.negative_aesthetic_score if args.negative_aesthetic_score else 2.5,
    }
    sampler, num_rows, num_cols = init_sampling(
        sampler=args.sampler,
        num_cols=args.num_cols,
        guider=args.guider,
        discretization=args.discretization,
        steps=args.sample_step,
        stage2strength=stage2strength,
    )
    num_samples = num_rows * num_cols

    print("Img2Img Sampling")
    s_time = time.time()
    out = model.do_sample(
        sampler,
        value_dict,
        num_samples,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=return_latents,
        filter=filter,
        amp_level=amp_level,
    )
    print(f"Img2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")

    return out


def sample(args):
    config = OmegaConf.load(args.config)
    config_base = OmegaConf.load(config.pop("base", ""))
    config_base.merge_with(config)
    config = config_base

    version = config.pop("version", "SDXL-base-1.0")
    version_dict = VERSION2SPECS.get(version)

    seed_everything(args.seed)

    # overwrite the ip_scale in config file
    config.model.params.network_config.params.ip_scale = args.ip_scale

    # Init Model
    model, filter = create_model(
        config,
        checkpoints=args.weight.split(","),
        freeze=True,
        load_filter=False,
        param_fp16=False,
        amp_level=args.ms_amp_level,
    )  # TODO: Add filter support

    save_path = os.path.join(args.save_path, version)
    is_legacy = True  # to be consistent with IP Adapter
    args.negative_prompt = args.negative_prompt if is_legacy else ""

    out = run_text2img(
        args,
        model,
        version_dict,
        is_legacy=is_legacy,
        filter=filter,
        stage2strength=None,
        amp_level=args.ms_amp_level,
    )

    out = out if isinstance(out, (tuple, list)) else [out, None]
    (samples, samples_z) = out

    perform_save_locally(save_path, samples)


if __name__ == "__main__":
    parser = get_parser_sample()
    args, _ = parser.parse_known_args()
    ms.context.set_context(
        mode=args.ms_mode, device_target=args.device_target, ascend_config=dict(precision_mode="must_keep_origin_dtype")
    )
    sample(args)
