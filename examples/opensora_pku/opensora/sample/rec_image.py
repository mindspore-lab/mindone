import argparse
import logging
import os
import sys

import cv2
import numpy as np
from albumentations import Compose, Lambda, Resize, ToFloat
from PIL import Image

import mindspore as ms

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)

from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

sys.path.append(".")

from opensora.models.causalvideovae import ae_wrapper
from opensora.npu_config import npu_config
from opensora.utils.utils import get_precision

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
    short_size = args.short_size
    npu_config.set_npu_env(args)
    npu_config.print_ops_dtype_info()

    set_logger(name="", output_dir=args.output_path, rank=0)
    dtype = get_precision(args.precision)
    if args.ms_checkpoint is not None and os.path.exists(args.ms_checkpoint):
        logger.info(f"Run inference with MindSpore checkpoint {args.ms_checkpoint}")
        state_dict = ms.load_checkpoint(args.ms_checkpoint)

        state_dict = dict(
            [k.replace("autoencoder.", "") if k.startswith("autoencoder.") else k, v] for k, v in state_dict.items()
        )
        state_dict = dict([k.replace("_backbone.", "") if "_backbone." in k else k, v] for k, v in state_dict.items())
    else:
        state_dict = None
    kwarg = {
        "state_dict": state_dict,
        "use_safetensors": True,
        "dtype": dtype,
    }
    vae = ae_wrapper[args.ae](args.ae_path, **kwarg)

    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor

    vae.set_train(False)
    for param in vae.get_parameters():
        param.requires_grad = False

    input_x = np.array(Image.open(image_path))  # (h w c)
    assert input_x.shape[2], f"Expect the input image has three channels, but got shape {input_x.shape}"
    x_vae = preprocess(input_x, short_size, short_size)  # use image as a single-frame video

    x_vae = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    latents = vae.encode(x_vae)
    latents = latents.to(dtype)
    image_recon = vae.decode(latents)  # b t c h w

    save_fp = os.path.join(args.output_path, args.rec_path)
    x = image_recon[0, 0, :, :, :]
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
    parser.add_argument("--ae", type=str, default="WFVAEModel_D8_4x8x8", choices=ae_wrapper.keys())
    parser.add_argument("--ae_path", type=str, default="results/pretrained")
    parser.add_argument("--ms_checkpoint", type=str, default=None)
    parser.add_argument("--short_size", type=int, default=336)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--tile_sample_min_size", type=int, default=256)
    parser.add_argument("--enable_tiling", action="store_true")
    # ms related
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
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
