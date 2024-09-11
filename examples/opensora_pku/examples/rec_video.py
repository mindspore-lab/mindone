"""
Run causal vae reconstruction on a given video.
Usage example:
python examples/rec_video.py \
    --ae_path path/to/vae/ckpt \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --sample_rate 1 \
    --num_frames 65 \
    --height 480 \
    --width 640 \
"""
import argparse
import importlib
import logging
import os
import random
import sys

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

import mindspore as ms
from mindspore import nn

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.visualize.videos import save_videos

sys.path.append(".")
from functools import partial

import cv2
from albumentations import Compose, Lambda, Resize, ToFloat
from opensora.dataset.transform import center_crop_th_tw
from opensora.models import CausalVAEModelWrapper
from opensora.models.causalvideovae.model.modules.updownsample import TrilinearInterpolate
from opensora.utils.ms_utils import init_env
from opensora.utils.utils import get_precision

logger = logging.getLogger(__name__)


def read_video(video_path: str, num_frames: int, sample_rate: int) -> ms.Tensor:
    decord_vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(decord_vr)
    sample_frames_len = sample_rate * num_frames

    if total_frames > sample_frames_len:
        s = random.randint(0, total_frames - sample_frames_len - 1)
        s = 0
        e = s + sample_frames_len
        num_frames = num_frames
    else:
        s = 0
        e = total_frames
        num_frames = int(total_frames / sample_frames_len * num_frames)
        print(
            f"sample_frames_len {sample_frames_len}, only can sample {num_frames * sample_rate}",
            video_path,
            total_frames,
        )

    frame_id_list = np.linspace(s, e - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    return video_data


def create_transform(max_height, max_width, num_frames):
    norm_fun = lambda x: 2.0 * x - 1.0

    def norm_func_albumentation(image, **kwargs):
        return norm_fun(image)

    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    targets = {"image{}".format(i): "image" for i in range(num_frames)}
    resize = [
        Lambda(
            name="crop_centercrop",
            image=partial(center_crop_th_tw, th=max_height, tw=max_width, top_crop=False),
            p=1.0,
        ),
        Resize(max_height, max_width, interpolation=mapping["bilinear"]),
    ]

    transform = Compose(
        [*resize, ToFloat(255.0), Lambda(name="ae_norm", image=norm_func_albumentation, p=1.0)],
        additional_targets=targets,
    )
    return transform


def preprocess(video_data, height: int = 128, width: int = 128):
    num_frames = video_data.shape[0]
    video_transform = create_transform(height, width, num_frames=num_frames)

    inputs = {"image": video_data[0]}
    for i in range(num_frames - 1):
        inputs[f"image{i}"] = video_data[i + 1]

    video_outputs = video_transform(**inputs)
    video_outputs = np.stack(list(video_outputs.values()), axis=0)  # (t h w c)
    # (t h w c) -> (c t h w)
    video_outputs = np.transpose(video_outputs, (3, 0, 1, 2))
    return video_outputs


def transform_to_rgb(x, rescale_to_uint8=True):
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    if rescale_to_uint8:
        x = (255 * x).astype(np.uint8)
    return x


def main(args):
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
        if args.save_memory:
            vae.vae.tile_sample_min_size = args.tile_sample_min_size
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8

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

    x_vae = preprocess(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
    dtype = get_precision(args.precision)
    x_vae = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    latents = vae.encode(x_vae)
    latents = latents.to(dtype)
    video_recon = vae.decode(latents)  # b t c h w

    save_fp = os.path.join(args.output_path, args.rec_path)
    if ".avi" in os.path.basename(save_fp):
        save_fp = save_fp.replace(".avi", ".mp4")
    if video_recon.shape[1] == 1:
        x = video_recon[0, 0, :, :, :].squeeze().to(ms.float32).asnumpy()
        original_rgb = x_vae[0, 0, :, :, :].squeeze().to(ms.float32).asnumpy()
        x = transform_to_rgb(x).transpose(1, 2, 0)  # c h w -> h w c
        original_rgb = transform_to_rgb(original_rgb).transpose(1, 2, 0)  # c h w -> h w c

        image = Image.fromarray(np.concatenate([x, original_rgb], axis=1) if args.grid else x)
        save_fp = save_fp.replace("mp4", "jpg")
        image.save(save_fp)
    else:
        save_video_data = video_recon.transpose(0, 1, 3, 4, 2).to(ms.float32).asnumpy()  # (b t c h w) -> (b t h w c)
        save_video_data = transform_to_rgb(save_video_data, rescale_to_uint8=False)
        original_rgb = transform_to_rgb(x_vae.to(ms.float32).asnumpy(), rescale_to_uint8=False).transpose(
            0, 2, 3, 4, 1
        )  # (b c t h w) -> (b t h w c)
        save_video_data = np.concatenate([original_rgb, save_video_data], axis=3) if args.grid else save_video_data
        save_videos(save_video_data, save_fp, loop=0, fps=args.fps)
    if args.grid:
        logger.info(f"Save original vs. reconstructed data to {save_fp}")
    else:
        logger.info(f"Save reconstructed data to {save_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--rec_path", type=str, default="")
    parser.add_argument("--ae_path", type=str, default="results/pretrained")
    parser.add_argument("--ms_checkpoint", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--tile_sample_min_size", type=int, default=256)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--save_memory", action="store_true")
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
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference, better to set to True when training vae",
    )
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
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
