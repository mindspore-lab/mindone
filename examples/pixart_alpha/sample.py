#!/usr/bin/env python
import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import tqdm
import yaml
from dataset import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from dataset.utils import classify_height_width_bin
from PIL import Image
from utils.model_utils import check_cfgs_in_parser, count_params, load_ckpt_params, remove_pname_prefix, str2bool
from utils.plot import image_grid, resize_and_crop_tensor

import mindspore as ms
from mindspore import ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from modules.pixart import PixArt_XL_2, PixArtMS_XL_2
from modules.text_encoder import T5Embedder
from modules.vae import SD_CONFIG, AutoencoderKL
from pipelines.infer_pipeline import PixArtInferPipeline

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args) -> None:
    set_random_seed(args.seed)
    ms.set_context(mode=args.mode, device_target=args.device_target)


def parse_args():
    parser = argparse.ArgumentParser(description="PixArt-Alpha Image generation")
    parser.add_argument(
        "-c",
        "--config",
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=512,
        help="image height",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=512,
        help="image width",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=64,
        choices=[128, 64, 32],
        help="network sample size",
    )
    parser.add_argument(
        "--use_resolution_binning",
        type=str2bool,
        default=True,
        help="If set to `True`, the requested height and width are first mapped to the closest resolutions of the bins",
    )
    parser.add_argument("--clean_caption", type=str2bool, default=False, help="clean the prompt before encoding.")
    parser.add_argument(
        "--checkpoint", default="models/PixArt-XL-2-512x512.ckpt", help="the path to the PixArt checkpoint."
    )
    parser.add_argument(
        "--vae_checkpoint",
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--t5_root", default="models/t5-v1_1-xxl", help="Path storing the T5 checkpoint and tokenizer configure file."
    )
    parser.add_argument("--t5_max_length", type=int, default=120, help="T5's embedded sequence length.")
    parser.add_argument(
        "--prompt", default="A small cactus with a happy face in the Sahara desert.", help="Prompt for sampling."
    )
    parser.add_argument("--negative_prompt", default="", help="Negative prompt for sampling.")
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion network."
    )
    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="the scale for classifier-free guidance")
    # MS new args
    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"], help="Device target")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=42, help="Inference seed")

    parser.add_argument(
        "--enable_flash_attention",
        default=True,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for PixArt. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument("--ddim_sampling", type=str2bool, default=True, help="Whether to use DDIM for sampling")
    parser.add_argument("--imagegrid", default=False, type=str2bool, help="Save the image in image-grids format.")
    parser.add_argument("--nrows", default=1, type=int, help="Number of rows in sampling (number of trials)")
    parser.add_argument("--ncols", default=1, type=int, help="Number of cols in sampling (batch size)")
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    args = parse_args()
    init_env(args)

    # 1.1. bin the size if need
    if args.use_resolution_binning:
        if args.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif args.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif args.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        else:
            raise ValueError("Invalid sample size")
        orig_height, orig_width = args.image_height, args.image_width
        height, width = classify_height_width_bin(orig_height, orig_width, ratios=aspect_ratio_bin)
    else:
        height, width = args.image_height, args.image_width

    # 2. network initiate and weight loading
    logger.info(f"{width}x{height} init")

    latent_height, latent_width = height // 8, width // 8

    if args.sample_size == 128:
        network = PixArtMS_XL_2(block_kwargs={"enable_flash_attention": args.enable_flash_attention})
    else:
        network = PixArt_XL_2(
            input_size=args.sample_size, block_kwargs={"enable_flash_attention": args.enable_flash_attention}
        )

    if args.dtype == "fp16":
        model_dtype = ms.float16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    try:
        network = load_ckpt_params(network, args.checkpoint)
    except Exception:
        param_dict = ms.load_checkpoint(args.checkpoint)
        param_dict = remove_pname_prefix(param_dict, prefix="network.")
        network = load_ckpt_params(network, param_dict)

    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(SD_CONFIG, 4, ckpt_path=args.vae_checkpoint)
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    # 2.3
    logger.info("text encoder init")
    text_encoder = T5Embedder(
        args.t5_root,
        use_text_preprocessing=args.clean_caption,
        model_max_length=args.t5_max_length,
        pretrained_ckpt=os.path.join(args.t5_root, "model.ckpt"),
    )

    # 3. build inference pipeline
    model_config = dict(
        height=height,
        width=width,
        sample_size=args.sample_size,
        hidden_size=network.hidden_size,
        patch_size=network.patch_size,
        lewei_scale=2.0 if args.sample_size == 128 else 1.0,
    )
    pipeline = PixArtInferPipeline(
        network,
        vae,
        text_encoder,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=args.ddim_sampling,
        multi_scale=args.sample_size == 128,
        model_config=model_config,
    )

    # 4. print key info
    num_params_vae, _ = count_params(vae)
    num_params_network, _ = count_params(network)
    num_params_text_encoder, _ = count_params(text_encoder)
    num_params = num_params_vae + num_params_network + num_params_text_encoder
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Num params: {num_params:,} (network: {num_params_network:,}, vae: {num_params_vae:,}, text_encoder: {num_params_text_encoder:,})",
            f"Use network dtype: {model_dtype}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    start_time = time.time()

    # infer
    x_samples = list()
    for _ in tqdm.trange(args.nrows):
        # Create sampling noise
        z = ops.randn((args.ncols, 4, latent_height, latent_width), dtype=ms.float32)
        y = args.prompt
        y_null = args.negative_prompt

        # init inputs
        inputs = dict(noise=z, y=y, y_null=y_null, scale=args.guidance_scale)

        output = pipeline(inputs).asnumpy()
        x_samples.append(output)

    x_samples = np.concatenate(x_samples, axis=0)
    end_time = time.time()

    if args.use_resolution_binning:
        x_samples = resize_and_crop_tensor(x_samples, orig_width, orig_height)

    # save result
    if not args.imagegrid:
        for i in range(x_samples.shape[0]):
            save_fp = f"{save_dir}/{i}.png"
            img = Image.fromarray((x_samples[i] * 255).astype(np.uint8))
            img.save(save_fp)
            logger.info(f"save to {save_fp}")
    else:
        save_fp = f"{save_dir}/sample.png"
        img = image_grid(x_samples, ncols=args.ncols)
        img.save(save_fp)
        logger.info(f"save to {save_fp}")
