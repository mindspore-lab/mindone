#!/usr/bin/env python
"""
IPAdapter SD image to image generation (Inpainting condition)
"""
import argparse
import logging
import os
import sys
import time
from typing import Tuple

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor

import mindspore as ms
import mindspore.ops as ops

sys.path.append("../stable_diffusion_v2/")
sys.path.append("../stable_diffusion_xl/")

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.logger import set_logger
from ldm.modules.train.tools import set_random_seed
from ldm.util import instantiate_from_config
from utils import model_utils

logger = logging.getLogger("image_inpainting")

# naming: {sd_base_version}-{variation}
_version_cfg = {
    "1.5": ("sd_models/merged/sd_v1.5_ip_adapter.ckpt", "inference/sd_v15.yaml", 512),
}


def load_model_from_config(config, ckpt):
    model = instantiate_from_config(config.model)

    def _load_model(_model, ckpt_fp, verbose=True, filter=None):
        if os.path.exists(ckpt_fp):
            param_dict = ms.load_checkpoint(ckpt_fp)
            if param_dict:
                param_not_load, ckpt_not_load = model_utils.load_param_into_net_with_filter(
                    _model, param_dict, filter=filter
                )
                if verbose:
                    if len(param_not_load) > 0:
                        logger.info(
                            "Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")])
                        )
                    if len(ckpt_not_load) > 0:
                        logger.info(
                            "Ckpt params not loaded: {}".format([p for p in ckpt_not_load if not p.startswith("adam")])
                        )
        else:
            raise FileNotFoundError(f"{ckpt_fp} doesn't exist")

    logger.info(f"Loading model from {ckpt}")
    _load_model(model, ckpt)

    model.set_train(False)
    model.set_grad(False)
    for param in model.get_parameters():
        param.requires_grad = False

    return model


def _check_strength(val: float):
    if val <= 0 or val > 1:
        raise ValueError("`strength` must be in range (0, 1]`")


def _cal_size(w: int, h: int, min_size: int = 512) -> Tuple[int, int]:
    if w < h:
        new_h = round(h / w * min_size)
        new_w = min_size
    else:
        new_w = round(w / h * min_size)
        new_h = min_size
    return new_w, new_h


def load_clip_image(image: str) -> ms.Tensor:
    image = Image.open(image)
    image = ImageOps.exif_transpose(image)  # type: Image.Image
    image = image.convert("RGB")
    image = CLIPImageProcessor()(image)
    image = ms.Tensor(image.pixel_values, ms.float32)
    return image


def load_ref_image(image: str, scale_factor: int = 8, min_size: int = 512) -> ms.Tensor:
    image = Image.open(image)
    image = ImageOps.exif_transpose(image)  # type: Image.Image
    image = image.convert("RGB")
    w, h = image.size
    w, h = _cal_size(w, h, min_size=min_size)
    w, h = (x - x % scale_factor for x in (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image, dtype=np.float32)
    image = image[None, ...]
    image = image.transpose(0, 3, 1, 2)
    image = image / 127.5 - 1.0
    image = ms.Tensor(image, ms.float32)
    return image


def load_ref_mask(mask: str, scale_factor: int = 8, min_size: int = 512) -> ms.Tensor:
    mask = Image.open(mask)
    mask = ImageOps.exif_transpose(mask)  # type: Image.Image
    mask = mask.convert("L")
    w, h = mask.size
    w, h = _cal_size(w, h, min_size=min_size)
    w, h = (x - x % scale_factor for x in (w, h))
    # downsample as vae does
    mask = mask.resize((w // scale_factor, h // scale_factor), resample=Image.NEAREST)
    mask = np.array(mask, dtype=np.float32)
    mask = mask[None, ...]
    mask = np.tile(mask, (1, 4, 1, 1))
    mask = mask / 255.0
    mask = 1 - mask  # keep black, inpaint white
    mask = ms.Tensor(mask, ms.float32)
    return mask


def main(args):
    # set logger
    set_logger(
        name="",
        output_dir=args.output_path,
        rank=0,
        log_level=eval(args.log_level),
    )

    work_dir = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"WORK DIR:{work_dir}")
    os.makedirs(args.output_path, exist_ok=True)
    outpath = args.output_path

    # read prompts
    batch_size = args.n_samples
    if not args.data_path:
        prompt = args.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        logger.info(f"Reading prompts from {args.data_path}")
        with open(args.data_path, "r") as f:
            prompts = f.read().splitlines()
            # TODO: try to put different prompts in a batch
            data = [batch_size * [prompt] for prompt in prompts]

    # read negative prompts
    if not args.negative_data_path:
        negative_prompt = args.negative_prompt
        assert negative_prompt is not None
        negative_data = [batch_size * [negative_prompt]]
    else:
        logger.info(f"Reading negative prompts from {args.negative_data_path}")
        with open(args.negative_data_path, "r") as f:
            negative_prompts = f.read().splitlines()
            # TODO: try to put different prompts in a batch
            negative_data = [batch_size * [negative_prompt] for negative_prompt in negative_prompts]

    # post-process negative prompts
    assert len(negative_data) <= len(data), "Negative prompts should be shorter than positive prompts"
    if len(negative_data) < len(data):
        logger.info("Negative prompts are shorter than positive prompts, padding blank prompts")
        blank_negative_prompt = batch_size * [""]
        for _ in range(len(data) - len(negative_data)):
            negative_data.append(blank_negative_prompt)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.ms_mode,
        device_target="Ascend",
        device_id=device_id,
        ascend_config=dict(precision_mode="must_keep_origin_dtype"),
    )

    set_random_seed(args.seed)

    # create model
    config = OmegaConf.load(args.config)
    config_base = OmegaConf.load(config.pop("base", ""))
    config_base.merge_with(config)
    config = config_base

    model = load_model_from_config(
        config,
        ckpt=args.ckpt_path,
    )

    # read image
    clip_img = load_clip_image(args.img)
    ref_img = load_ref_image(args.ref_img)
    ref_mask = load_ref_mask(args.ref_mask)

    # get ref image latent
    ref_img_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_img))
    ref_img_latent = ops.tile(ref_img_latent, (batch_size, 1, 1, 1))

    # mask
    ref_mask = ops.tile(ref_mask, (batch_size, 1, 1, 1))

    # get image conditioning
    clip_img_c = model.embedder(clip_img)
    clip_img_c = ops.tile(clip_img_c, (batch_size, 1, 1))

    clip_img_uc = model.embedder()
    clip_img_uc = ops.tile(clip_img_uc, (batch_size, 1, 1))

    prediction_type = getattr(config.model, "prediction_type", "noise")
    logger.info(f"Prediction type: {prediction_type}")

    # create sampler (please refer to sd_v2 for more sampler usage)
    sampler = DDIMSampler(model)
    sname = "ddim"

    # determine total timesteps for q_forward
    sampler.make_schedule(args.sampling_steps, verbose=False)  # create the ddim steps internally
    sampling_steps = round(args.sampling_steps * args.strength)  # the real sampling steps
    if sampling_steps < 1:
        raise RuntimeError("`strength` must be large enougth such that at least one sampling step can be performed.")
    start_sample_step = args.sampling_steps - sampling_steps
    n_timestep = sampler.ddim_timesteps[::-1][None, start_sample_step, ...]

    if sampling_steps == args.sampling_steps:
        timesteps = None
    else:
        timesteps = sampling_steps + 1

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            "Distributed mode: False",
            f"Number of input prompts: {len(data)}",
            f"Number of input negative prompts: {len(negative_data)}",
            f"Number of trials for each prompt: {args.n_iter}",
            f"Number of samples in each trial: {args.n_samples}",
            f"Model: StableDiffusion v-{args.version}",
            f"Precision: {model.model.diffusion_model.dtype}",
            f"Pretrained ckpt path: {args.ckpt_path}",
            f"Sampler: {sname}",
            f"Sampling steps: {sampling_steps} ({args.sampling_steps} * {args.strength})",
            f"Uncondition guidance scale: {args.scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    # infer
    all_samples = list()
    for i, prompts in enumerate(data):
        negative_prompts = negative_data[i]
        logger.info(
            f"[{i + 1}/{len(data)}] Generating images with conditions:\nPrompt(s): {prompts[0]}\n"
            f"Negative prompt(s): {negative_prompts[0]}"
        )
        for n in range(args.n_iter):
            start_time = time.time()
            uc = None
            if args.scale != 1.0:
                if isinstance(negative_prompts, tuple):
                    negative_prompts = list(negative_prompts)
                tokenized_negative_prompts = model.tokenize(negative_prompts)
                uc = model.get_learned_conditioning(tokenized_negative_prompts)
                # concat text/img embedding
                uc = ops.concat([uc.to(ms.float32), clip_img_uc.to(ms.float32)], axis=1)
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            tokenized_prompts = model.tokenize(prompts)
            c = model.get_learned_conditioning(tokenized_prompts)
            # concat text/img embedding
            c = ops.concat([c.to(ms.float32), clip_img_c.to(ms.float32)], axis=1)

            noise = ops.randn(ref_img_latent.shape)
            start_code = model.q_sample(ref_img_latent, n_timestep, noise)
            shape = [4, ref_img_latent.shape[2], ref_img_latent.shape[3]]
            samples_ddim, _ = sampler.sample(
                S=args.sampling_steps,
                conditioning=c,
                batch_size=args.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=args.scale,
                unconditional_conditioning=uc,
                eta=args.ddim_eta,
                x_T=start_code,
                timesteps=timesteps,
                mask=ref_mask,
                x0=ref_img_latent,
                noise=noise,
            )
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = ops.clip_by_value((x_samples_ddim + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
            x_samples_ddim_numpy = x_samples_ddim.asnumpy()

            if not args.skip_save:
                for x_sample in x_samples_ddim_numpy:
                    x_sample = 255.0 * x_sample.transpose(1, 2, 0)
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1

            if not args.skip_grid:
                all_samples.append(x_samples_ddim_numpy)

            end_time = time.time()
            logger.info(
                f"{batch_size * (n + 1)}/{batch_size * args.n_iter} images generated, "
                f"time cost for current trial: {end_time - start_time:.3f}s"
            )

    logger.info(f"Done! All generated images are saved in: {outpath}/samples" f"\nEnjoy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="?",
        default="",
        help="path to a file containing prompt list (each line in the file corresponds to a prompt to render).",
    )
    parser.add_argument("--img", required=True, help="Path of the image input.")
    parser.add_argument(
        "--negative_data_path",
        type=str,
        nargs="?",
        default="",
        help="path to a file containing negative prompt list (each line in the file corresponds to a prompt not to "
        "render).",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="1.5",
        help="Stable diffusion version. Options: '1.5'",
    )
    parser.add_argument(
        "--prompt", type=str, nargs="?", default="best quality, high quality", help="the prompt to render"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        nargs="?",
        default="monochrome, lowres, bad anatomy, worst quality, low quality",
        help="the negative prompt not to render",
    )
    parser.add_argument("--output_path", type=str, nargs="?", default="outputs/demo/SD", help="dir to write results to")
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps.",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="number of iterations or trials. sample this often, ",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config which constructs model. If None, select by version",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--strength", type=float, default=0.7, help="the extent to transform the reference image")
    parser.add_argument("--ref_img", required=True, help="Path of the reference image")
    parser.add_argument("--ref_mask", required=True, help="Path of the mask of reference image")
    args = parser.parse_args()

    # check args
    if args.version:
        if args.version not in _version_cfg:
            raise ValueError(f"Unknown version: {args.version}. Supported SD versions are: {list(_version_cfg.keys())}")

    if args.ckpt_path is None:
        ckpt_name = _version_cfg[args.version][0]
        args.ckpt_path = "checkpoints/" + ckpt_name

    if args.config is None:
        args.config = os.path.join("configs", _version_cfg[args.version][1])

    if args.scale is None:
        args.scale = 9.0 if args.version.startswith("2.") else 7.5

    _check_strength(args.strength)

    # core task
    main(args)
