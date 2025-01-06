"""
Run causal vae reconstruction on a given image
Usage example:
python examples/rec_image.py \
    --image_path test.jpg \
    --rec_path rec.jpg \
    --image_size 512 \
"""
import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

sys.path.append(".")

import cv2
from albumentations import Compose, Lambda, Resize, ToFloat
from hyvideo.constants import PRECISION_TO_TYPE, PRECISIONS, VAE_PATH
from hyvideo.utils.ms_utils import init_env
from hyvideo.vae import load_vae
from hyvideo.vae.unet_causal_3d_blocks import GroupNorm, MSInterpolate

logger = logging.getLogger(__name__)


def create_transform(max_height, max_width):
    norm_fun = lambda x: 2.0 * x - 1.0

    def norm_func_albumentation(image, **kwargs):
        return norm_fun(image)

    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    resize = [
        Resize(max_height, max_width, interpolation=mapping["bilinear"]),
    ]

    transform = Compose(
        [*resize, ToFloat(255.0), Lambda(name="ae_norm", image=norm_func_albumentation, p=1.0)],
    )
    return transform


def preprocess(image, height: int = 128, width: int = 128):
    video_transform = create_transform(height, width)

    image = video_transform(image=image)["image"]  # (h w c)
    # (h w c) -> (c h w) -> (c t h w)
    image = np.transpose(image, (2, 0, 1))[:, None, :, :]
    return image


def transform_to_rgb(x, rescale_to_uint8=True):
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    if rescale_to_uint8:
        x = (255 * x).astype(np.uint8)
    return x


def main(args):
    image_path = args.image_path
    image_size = args.image_size
    init_env(
        mode=args.mode,
        device_target=args.device,
        precision_mode=args.precision_mode,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
    )

    set_logger(name="", output_dir=args.output_path, rank=0)

    if args.ms_checkpoint is not None and os.path.exists(args.ms_checkpoint):
        logger.info(f"Run inference with MindSpore checkpoint {args.ms_checkpoint}")
        state_dict = ms.load_checkpoint(args.ms_checkpoint)
        # rm 'network.' prefix
        state_dict = dict(
            [k.replace("autoencoder.", "") if k.startswith("autoencoder.") else k, v] for k, v in state_dict.items()
        )
    else:
        state_dict = None

    vae, _, s_ratio, t_ratio = load_vae(
        args.vae,
        args.vae_precision,
        logger=logger,
        state_dict=state_dict,
    )
    # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    if args.vae_tiling:
        vae.enable_tiling()
        # vae.tile_overlap_factor = args.tile_overlap_factor
    if args.precision in ["fp16", "bf16"]:
        amp_level = "O2"
        dtype = PRECISION_TO_TYPE[args.precision]
        if dtype == ms.float16:
            custom_fp32_cells = [GroupNorm] if args.vae_keep_gn_fp32 else []
        else:
            custom_fp32_cells = [nn.ReplicationPad3d, MSInterpolate]

        vae = auto_mixed_precision(vae, amp_level, dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(
            f"Set mixed precision to {amp_level} with dtype={args.precision}, custom fp32_cells {custom_fp32_cells}"
        )
    elif args.precision == "fp32":
        dtype = PRECISION_TO_TYPE[args.precision]
    else:
        raise ValueError(f"Unsupported precision {args.precision}")
    input_x = np.array(Image.open(image_path))  # (h w c)
    assert input_x.shape[2], f"Expect the input image has three channels, but got shape {input_x.shape}"
    x_vae = preprocess(input_x, image_size, image_size)  # use image as a single-frame video
    x_vae = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    latents = vae.encode(x_vae)
    latents = latents.to(dtype)
    image_recon = vae.decode(latents)  # b c t h w

    save_fp = os.path.join(args.output_path, args.rec_path)
    x = image_recon[0, :, 0, :, :]
    x = x.squeeze().asnumpy()
    x = transform_to_rgb(x)
    x = x.transpose(1, 2, 0)
    if args.grid:
        x = np.concatenate([input_x, x], axis=1)
    image = Image.fromarray(x)
    image.save(save_fp)
    if args.grid:
        logger.info(f"Save original vs. reconstructed data to {save_fp}")
    else:
        logger.info(f"Save reconstructed data to {save_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--rec_path", type=str, default="")
    # - VAE
    parser.add_argument(
        "--vae",
        type=str,
        default="884-16c-hy",
        choices=list(VAE_PATH),
        help="Name of the VAE model.",
    )
    parser.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model.",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable tiling for the VAE model to save GPU memory.",
    )

    parser.add_argument("--ms_checkpoint", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=336)

    # ms related
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="mixed precision type, if fp32, all layer precision is float32 (amp_level=O0),  \
                if bf16 or fp16, amp_level==O2, part of layers will compute in bf16 or fp16 such as matmul, dense, conv.",
    )
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference, better to set to True when training vae",
    )
    parser.add_argument(
        "--output_path", default="samples/vae_recons", type=str, help="output directory to save inference results"
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="whether to use grid to show original and reconstructed data",
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument(
        "--jit_syntax_level", default="strict", choices=["strict", "lax"], help="Set jit syntax level: strict or lax"
    )
    args = parser.parse_args()
    main(args)
