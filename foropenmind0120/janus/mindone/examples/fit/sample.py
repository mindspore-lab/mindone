#!/usr/bin/env python
"""
FiT inference pipeline
"""
import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import yaml
from PIL import Image
from utils.model_utils import check_cfgs_in_parser, count_params, load_fit_ckpt_params, str2bool
from utils.plot import image_grid

import mindspore as ms
from mindspore import Tensor, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from modules.autoencoder import SD_CONFIG, AutoencoderKL
from pipelines.infer_pipeline import FiTInferPipeline

from mindone.models.fit import FiT_models
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode, device_target=args.device_target, device_id=device_id, jit_config={"jit_level": args.jit_level}
    )

    return device_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/inference/fit-xl-2-256x256.yaml",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="image height",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="image width",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="image size",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="FiT-XL/2",
        help="Model name , such as FiT-XL/2",
    )
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size")
    parser.add_argument("--embed_dim", type=int, default=72, help="Embed Dim")
    parser.add_argument("--embed_method", default="rotate", help="Embed Method")
    parser.add_argument("--fit_checkpoint", type=str, required=True, help="the path to the FiT checkpoint.")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="the scale for classifier-free guidance")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--use_fp16",
        default=True,
        type=str2bool,
        help="whether to use fp16 for FiT mode. Default is True",
    )
    parser.add_argument("--ddim_sampling", type=str2bool, default=True, help="Whether to use DDIM for sampling")
    parser.add_argument("--imagegrid", default=False, type=str2bool, help="Save the image in image-grids format.")
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
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
    set_random_seed(args.seed)

    # 2. model initiate and weight loading
    # 2.1 fit
    logger.info(f"{args.model_name}-{args.image_width}x{args.image_height} init")
    latent_height, latent_width = args.image_height // 8, args.image_width // 8
    fit_model = FiT_models[args.model_name](
        num_classes=1000,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        pos=args.embed_method,
    )

    if args.use_fp16:
        fit_model = auto_mixed_precision(fit_model, amp_level="O2")

    fit_model = load_fit_ckpt_params(fit_model, args.fit_checkpoint)
    fit_model = fit_model.set_train(False)
    for param in fit_model.get_parameters():  # freeze fit_model
        param.requires_grad = False

    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,  # disable amp for vae
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # Create sampling noise:
    n = len(class_labels)
    z = ops.randn((n, 4, latent_height, latent_width), dtype=ms.float32)
    y = Tensor(class_labels)
    y_null = ops.ones_like(y) * 1000

    model_config = dict(
        C=4,
        max_size=args.image_size // 8,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        embed_method=args.embed_method,
        max_length=args.image_size * args.image_size // 8 // 8 // args.patch_size // args.patch_size,
    )

    # 3. build inference pipeline
    pipeline = FiTInferPipeline(
        fit_model,
        vae,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
        ddim_sampling=args.ddim_sampling,
        model_config=model_config,
    )

    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_fit, num_params_fit_trainable = count_params(fit_model)
    num_params = num_params_vae + num_params_fit
    num_params_trainable = num_params_vae_trainable + num_params_fit_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Class labels: {class_labels}",
            f"Num params: {num_params:,} (fit: {num_params_fit:,}, vae: {num_params_vae:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"Use FP16: {args.use_fp16}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)
    # init inputs
    inputs = {}
    inputs["noise"] = z
    inputs["y"] = y
    inputs["y_null"] = y_null
    inputs["scale"] = args.guidance_scale

    logger.info(f"Sampling class labels: {class_labels}")
    start_time = time.time()

    # infer
    x_samples = pipeline(inputs)
    x_samples = x_samples.asnumpy()

    end_time = time.time()

    # save result
    if not args.imagegrid:
        for i, class_label in enumerate(class_labels, 0):
            save_fp = f"{save_dir}/class-{class_label}.png"
            img = Image.fromarray((x_samples[i] * 255).astype(np.uint8))
            img.save(save_fp)
            logger.info(f"save to {save_fp}")
    else:
        save_fp = f"{save_dir}/sample.png"
        img = image_grid(x_samples)
        img.save(save_fp)
        logger.info(f"save to {save_fp}")
