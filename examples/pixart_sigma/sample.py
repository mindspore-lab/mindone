#!/usr/bin/env python
import argparse
import logging
import os
import sys

import numpy as np
import tqdm
import yaml
from PIL import Image

import mindspore as ms
import mindspore.ops as ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from pixart.dataset import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_2048_BIN,
    classify_height_width_bin,
)
from pixart.modules.pixart import PixArt_XL_2, PixArtMS_XL_2
from pixart.pipelines.infer_pipeline import PixArtInferPipeline
from pixart.utils import (
    check_cfgs_in_parser,
    count_params,
    image_grid,
    load_ckpt_params,
    resize_and_crop_tensor,
    str2bool,
)
from transformers import AutoTokenizer

from mindone.diffusers import AutoencoderKL
from mindone.transformers import T5EncoderModel
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args) -> None:
    set_random_seed(args.seed)
    ms.set_context(mode=args.mode, device_target=args.device_target, jit_config=dict(jit_level=args.jit_level))


def parse_args():
    parser = argparse.ArgumentParser(
        description="PixArt-Alpha Image generation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument("--output_path", default="./samples", help="Output directory to save the generated images.")
    parser.add_argument("--image_height", default=512, type=int, help="Target height of the generated image.")
    parser.add_argument("--image_width", default=512, type=int, help="Target width of the generated image.")
    parser.add_argument("--sample_size", default=64, type=int, choices=[128, 64, 32], help="Network sample size")
    parser.add_argument(
        "--use_resolution_binning",
        default=True,
        type=str2bool,
        help="If set to `True`, the requested height and width are first mapped to the closest resolutions of the bins",
    )
    parser.add_argument(
        "--checkpoint", default="models/PixArt-XL-2-512x512.ckpt", help="Path to the PixArt checkpoint."
    )
    parser.add_argument("--vae_root", default="models/vae", help="Path storing the VAE checkpoint and configure file.")
    parser.add_argument(
        "--tokenizer_root", default="models/tokenizer", help="Path storing the T5 checkpoint and configure file."
    )
    parser.add_argument(
        "--text_encoder_root", default="models/text_encoder", help="Path storing the T5 tokenizer and configure file."
    )

    parser.add_argument("--t5_max_length", default=300, type=int, help="T5's embedded sequence length.")
    parser.add_argument(
        "--prompt", default="A small cactus with a happy face in the Sahara desert.", help="Prompt for sampling."
    )
    parser.add_argument("--negative_prompt", default="", help="Negative prompt for sampling.")
    parser.add_argument(
        "--sd_scale_factor", default=0.13025, type=float, help="VAE scale factor of Stable Diffusion network."
    )
    parser.add_argument("--sampling_steps", default=50, type=int, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", default=4.5, type=float, help="Scale value for classifier-free guidance")

    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"], help="Device target")
    parser.add_argument(
        "--mode", default=1, choices=[0, 1], type=int, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1)"
    )
    parser.add_argument("--jit_level", default="O0", choices=["O0", "O1"], help="Jit Level")
    parser.add_argument("--seed", default=42, type=int, help="Inference seed")

    parser.add_argument(
        "--enable_flash_attention", default=True, type=str2bool, help="Whether to enable flash attention."
    )

    parser.add_argument("--kv_compress", default=False, type=str2bool, help="Do KV Compression if it is True.")
    parser.add_argument(
        "--kv_compress_sampling",
        default="conv",
        choices=["conv", "ave", "uniform"],
        help="Sampling method in KV compression.",
    )
    parser.add_argument("--kv_compress_scale_factor", default=1, type=int, help="Scaling value in KV compression.")
    parser.add_argument("--kv_compress_layer", nargs="*", type=int, help="Network layers performing KV compression.")

    parser.add_argument(
        "--dtype", default="fp16", choices=["bf16", "fp16", "fp32"], help="what data type to use for PixArt."
    )

    parser.add_argument("--ddim_sampling", default=True, type=str2bool, help="Whether to use DDIM for sampling")
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


def main(args):
    set_logger(output_dir="logs/sample")

    # 1. init env
    init_env(args)

    # 1.1. bin the size if need
    if args.use_resolution_binning:
        if args.sample_size == 256:
            aspect_ratio_bin = ASPECT_RATIO_2048_BIN
        elif args.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif args.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif args.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        else:
            raise ValueError(f"Invalid sample size: `{args.sample_size}`.")
        orig_height, orig_width = args.image_height, args.image_width
        height, width = classify_height_width_bin(orig_height, orig_width, ratios=aspect_ratio_bin)
    else:
        height, width = args.image_height, args.image_width

    # 2. network initiate and weight loading
    # 2.1 PixArt
    logger.info(f"{width}x{height} init")
    latent_height, latent_width = height // 8, width // 8

    pe_interpolation = args.sample_size / 64

    network_fn = PixArt_XL_2 if args.sample_size == 32 else PixArtMS_XL_2
    sampling = args.kv_compress_sampling if args.kv_compress else None
    network = network_fn(
        input_size=args.sample_size,
        pe_interpolation=pe_interpolation,
        model_max_length=args.t5_max_length,
        sampling=sampling,
        scale_factor=args.kv_compress_scale_factor,
        kv_compress_layer=args.kv_compress_layer,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
    )

    if args.dtype == "fp16":
        model_dtype = ms.float16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if args.checkpoint:
        network = load_ckpt_params(network, args.checkpoint)
    else:
        raise ValueError("`checkpoint` must be provided to run inference.")

    # 2.2 VAE
    logger.info("vae init")
    vae = AutoencoderKL.from_pretrained(args.vae_root)  # keep in fp32

    # 2.3 T5
    logger.info("text encoder init")
    text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root, model_max_length=args.t5_max_length)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_root)

    # 3. build inference pipeline
    pipeline = PixArtInferPipeline(
        network,
        vae,
        text_encoder,
        text_tokenizer,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=args.ddim_sampling,
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

    if args.use_resolution_binning:
        x_samples = resize_and_crop_tensor(x_samples, orig_width, orig_height)

    # save result
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    if not args.imagegrid:
        for i in range(x_samples.shape[0]):
            save_fp = os.path.join(args.output_path, f"{i}.png")
            img = Image.fromarray((x_samples[i] * 255).astype(np.uint8))
            img.save(save_fp)
            logger.info(f"save to {save_fp}")
    else:
        save_fp = os.path.join(args.output_path, "sample.png")
        img = image_grid(x_samples, ncols=args.ncols)
        img.save(save_fp)
        logger.info(f"save to {save_fp}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
