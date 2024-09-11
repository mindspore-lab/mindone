"""
Run causal vae reconstruction on a given image
Usage example:
python examples/rec_image.py \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae \
    --image_path test.jpg \
    --rec_path rec.jpg \
    --device Ascend \
    --short_size 512 \
    --enable_tiling
"""
import argparse
import importlib
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
from opensora.models import CausalVAEModelWrapper
from opensora.models.causalvideovae.model.modules.updownsample import TrilinearInterpolate
from opensora.utils.ms_utils import init_env
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
    image = np.transpose(image, (2, 1, 0))[:, None, :, :]
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
    else:
        # need torch installation to load from pt checkpoint!
        _torch_available = importlib.util.find_spec("torch") is not None
        if not _torch_available:
            logger.info(
                "Torch is not installed. Cannot load from torch checkpoint. Will search for safetensors under the given directory."
            )
            state_dict = None
        else:
            from opensora.utils.utils import load_torch_state_dict_to_ms_ckpt

            state_dict = load_torch_state_dict_to_ms_ckpt(os.path.join(args.ae_path, "checkpoint.ckpt"))
    kwarg = {"state_dict": state_dict}
    vae = CausalVAEModelWrapper(args.ae_path, **kwarg)

    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor

    vae.set_train(False)
    for param in vae.get_parameters():
        param.requires_grad = False
    if args.precision in ["fp16", "bf16"]:
        amp_level = "O2"
        dtype = get_precision(args.precision)
        if dtype == ms.float16:
            custom_fp32_cells = [nn.GroupNorm] if args.vae_keep_gn_fp32 else []
        else:
            custom_fp32_cells = [nn.AvgPool2d, TrilinearInterpolate]

        vae = auto_mixed_precision(vae, amp_level, dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(
            f"Set mixed precision to {amp_level} with dtype={args.precision}, custom fp32_cells {custom_fp32_cells}"
        )
    elif args.precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {args.precision}")
    input_x = np.array(Image.open(image_path))  # (h w c)
    assert input_x.shape[2], f"Expect the input image has three channels, but got shape {input_x.shape}"
    x_vae = preprocess(input_x, short_size, short_size)  # use image as a single-frame video
    dtype = get_precision(args.precision)
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
    parser.add_argument("--ae_path", type=str, default="results/pretrained")
    parser.add_argument("--ms_checkpoint", type=str, default=None)
    parser.add_argument("--short_size", type=int, default=336)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--tile_sample_min_size", type=int, default=256)
    parser.add_argument("--enable_tiling", action="store_true")
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
