import argparse
import datetime
import logging
import os
import sys
import time

import numpy as np
import yaml
from PIL import Image
from utils.model_utils import _check_cfgs_in_parser, load_dit_ckpt_params, str2bool

import mindspore as ms
from mindspore import Tensor, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from modules.autoencoder import SD_CONFIG, AutoencoderKL
from modules.dit.dit_models import DiT_models
from pipelines.infer_image_pipeline import DiTInferImage

from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.ms_mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.ms_mode,
        device_target=args.device_target,
        device_id=device_id,
    )

    return device_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="path to source torch checkpoint, which ends with .pt",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="DiT-XL/2",
        help="Model name , such as DiT-XL/2, DiT-L/2",
    )
    parser.add_argument(
        "--dit_checkpoint", type=str, default="models/DiT-XL-2-256x256.ckpt", help="the path to the DiT checkpoint."
    )
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
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether enable flash attention. Default is False",
    )
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
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
    # 2.1 dit
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    latent_size = args.image_size // 8
    dit_model = DiT_models[args.model_name](
        input_size=latent_size, num_classes=1000, block_kwargs={"enable_flash_attention": args.enable_flash_attention}
    )
    dit_model = load_dit_ckpt_params(dit_model, args.dit_checkpoint)

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
    z = ops.randn((n, 4, latent_size, latent_size), dtype=ms.float32)
    y = Tensor(class_labels)
    y_null = ops.ones_like(y) * dit_model.num_classes

    # 3. build inference pipeline
    pipeline = DiTInferImage(
        dit_model,
        vae,
        scale_factor=args.sd_scale_factor,
        num_inference_steps=args.sampling_steps,
        guidance_rescale=args.guidance_scale,
    )

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
    for i, class_label in enumerate(class_labels, 0):
        save_fp = f"{save_dir}/class-{class_label}.png"
        img = Image.fromarray((x_samples[i] * 255).astype(np.uint8))
        img.save(save_fp)
        logger.info(f"save to {save_fp}")
