#!/usr/bin/env python
"""
Insant ID Inference
"""
import argparse
import ast
import os
import sys
import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

import mindspore.ops as ops

sys.path.append("../ip_adapter/")
sys.path.append("../stable_diffusion_xl/")
sys.path.append("../stable_diffusion_v2/")

from gm.helpers import VERSION2SPECS, create_model, init_sampling, perform_save_locally
from gm.util import seed_everything
from insightface.app import FaceAnalysis
from instantid.util import draw_kps
from omegaconf import OmegaConf

import mindspore as ms


def get_parser_sample():
    parser = argparse.ArgumentParser(description="sampling with sd-xl")
    parser.add_argument(
        "--config", type=str, default="configs/inference/sd_xl_base_controlnet.yaml", help="Path of the config file"
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="checkpoints/merged/sd_xl_base_1.0_ms_instantid.ckpt",
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--ip_scale", type=float, default=0.8, help="IP Scale, control the attention strength of the image input."
    )
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--prompt", type=str, default="best quality, high quality", help="Prompt input")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, "
        "glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), "
        "watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured",
        help="Negative prompt input",
    )
    parser.add_argument("--img", type=str, required=True, help="Path of the input image file")
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
    parser.add_argument("--sample_step", type=int, default=30, help="Number of sampling steps")
    parser.add_argument("--num_cols", type=int, default=1, help="Number of images in single trial")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials.")
    parser.add_argument("--seed", type=int, default=42, help="Seed number")
    parser.add_argument("--save_path", type=str, default="outputs/demo/", help="Save directory")

    # system
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--ms_jit", type=ast.literal_eval, default=True, help="use jit or not")
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    return parser


def _cal_size(w: int, h: int, min_size: int = 512) -> Tuple[int, int]:
    if w < h:
        new_h = round(h / w * min_size)
        new_w = min_size
    else:
        new_w = round(w / h * min_size)
        new_h = min_size
    return new_w, new_h


def _process_image_emb(face_emb: np.ndarray) -> ms.Tensor:
    face_emb = face_emb[None, None, ...]
    face_emb = ms.Tensor(face_emb, dtype=ms.float32)
    return face_emb


def _process_control_image(image: Image.Image, scale_factor: int = 8, min_size: int = 1024) -> ms.Tensor:
    image = image.convert("RGB")
    w, h = image.size
    w, h = _cal_size(w, h, min_size=min_size)
    w, h = (x - x % scale_factor for x in (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image, dtype=np.float32)
    image = image[None, ...]
    image = image.transpose(0, 3, 1, 2)
    image = image / 255.0
    image = ms.Tensor(image, ms.float32)
    return image


def _resize_img(input_image, max_side=1280, min_side=1024, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / min(h, w)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)
    return input_image


def load_and_process_image(image: str) -> Image.Image:
    image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = _resize_img(image)
    # TODO: implement face model
    app = FaceAnalysis(name="antelopev2", root="./", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
    face_emb = face_info["embedding"]
    face_kps = draw_kps(image, face_info["kps"])

    face_emb = _process_image_emb(face_emb)
    face_kps = _process_control_image(face_kps)
    return face_emb, face_kps


def run_text2img(
    args,
    model,
    version_dict,
    save_path,
    is_legacy=False,
    return_latents=False,
    filter=None,
    stage2strength=None,
    amp_level="O0",
):
    C = version_dict["C"]
    F = version_dict["f"]

    img_emb, face_kps = load_and_process_image(args.img)
    face_kps = ops.tile(face_kps, (args.num_cols, 1, 1, 1))

    # TODO: for conditional guidance, move to internal
    face_kps = ops.concat([face_kps, face_kps], axis=0)
    _, _, H, W = face_kps.shape

    value_dict = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "clip_img": img_emb,
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
        guidance_scale=args.guidance_scale,
        discretization=args.discretization,
        steps=args.sample_step,
        stage2strength=stage2strength,
    )
    num_samples = num_rows * num_cols

    print("Img2Img Sampling")
    for _ in range(args.num_trials):
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
            control=face_kps,
        )
        print(f"Img2Img sample step {sampler.num_steps}, time cost: {time.time() - s_time:.2f}s")
        perform_save_locally(save_path, [out])


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

    run_text2img(
        args,
        model,
        version_dict,
        save_path,
        is_legacy=is_legacy,
        filter=filter,
        stage2strength=None,
        amp_level=args.ms_amp_level,
    )


if __name__ == "__main__":
    parser = get_parser_sample()
    args, _ = parser.parse_known_args()
    ms.set_context(
        mode=args.ms_mode, device_target=args.device_target, ascend_config=dict(precision_mode="must_keep_origin_dtype")
    )
    sample(args)
