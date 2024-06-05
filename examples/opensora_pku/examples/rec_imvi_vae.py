"""
Run causal vae reconstruction on a given video.
Usage example:
python examples/rec_imvi_vae.py \
    --model_path path/to/vae/ckpt \
    --video_path test.mp4 \
    --rec_path rec.mp4 \
    --sample_rate 1 \
    --num_frames 65 \
    --resolution 512 \
    --crop_size 512 \
    --ae CausalVAEModel_4x8x8 \
"""
import argparse
import logging
import os
import random
import sys

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

import mindspore as ms
from mindspore import nn, ops

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.visualize.videos import save_videos

sys.path.append(".")
from opensora.models.ae import getae_wrapper
from opensora.models.ae.videobase.modules.updownsample import TrilinearInterpolate

# from opensora.models.ae.videobase.causal_vae.modeling_causalvae import TimeDownsample2x, TimeUpsample2x
from opensora.utils.dataset_utils import create_video_transforms
from opensora.utils.utils import get_precision

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device,
        device_id=device_id,
    )
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})
    return device_id


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


def preprocess(video_data, resolution=128, crop_size=128):
    num_frames = video_data.shape[0]
    video_transform = create_video_transforms(
        resolution, crop_size, num_frames=num_frames, backend="al", disable_flip=True
    )

    inputs = {"image": video_data[0]}
    for i in range(num_frames - 1):
        inputs[f"image{i}"] = video_data[i + 1]

    video_outputs = video_transform(**inputs)
    video_outputs = np.stack(list(video_outputs.values()), axis=0)  # (t h w c)
    video_outputs = (video_outputs / 255.0) * 2 - 1.0
    # (t h w c) -> (c t h w)
    video_outputs = np.transpose(video_outputs, (3, 0, 1, 2))
    return video_outputs


def process_in_chunks(
    video_data: ms.Tensor,
    model: nn.Cell,
    chunk_size: int,
    overlap: int,
):
    assert (chunk_size + overlap - 1) % 4 == 0
    num_frames = video_data.shape[2]
    output_chunks = []

    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        if start + chunk_size + overlap < num_frames:
            end += overlap
        chunk = video_data[:, :, start:end, :, :]

        latents = model.encode(chunk)
        recon_chunk = model.decode(latents.to(ms.float16)).float()  # b t c h w
        recon_chunk = recon_chunk.permute(0, 2, 1, 3, 4)

        if output_chunks:
            overlap_step = min(overlap, recon_chunk.shape[2])
            overlap_tensor = output_chunks[-1][:, :, -overlap_step:] * 1 / 4 + recon_chunk[:, :, :overlap_step] * 3 / 4
            output_chunks[-1] = ops.cat((output_chunks[-1][:, :, :-overlap], overlap_tensor), axis=2)
            if end < num_frames:
                output_chunks.append(recon_chunk[:, :, overlap:])
            else:
                output_chunks.append(recon_chunk[:, :, :, :, :])
        else:
            output_chunks.append(recon_chunk)
        start += chunk_size
    return ops.cat(output_chunks, axis=2).permute(0, 2, 1, 3, 4)


def transform_to_rgb(x, rescale_to_uint8=True):
    x = np.clip(x, -1, 1)
    x = (x + 1) / 2
    if rescale_to_uint8:
        x = (255 * x).astype(np.uint8)
    return x


def main(args):
    init_env(args)
    set_logger(name="", output_dir=args.output_path, rank=0)

    kwarg = {}
    # vae = getae_wrapper(args.ae)(getae_model_config(args.ae), args.model_path, **kwarg)
    vae = getae_wrapper(args.ae)(args.model_path, **kwarg)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor

    vae.set_train(False)
    for param in vae.get_parameters():
        param.requires_grad = False
    if args.precision in ["fp16", "bf16"]:
        amp_level = "O2"
        dtype = get_precision(args.precision)
        custom_fp32_cells = [nn.GroupNorm] if dtype == ms.float16 else [nn.AvgPool2d, TrilinearInterpolate]
        vae = auto_mixed_precision(vae, amp_level, dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(f"Set mixed precision to O2 with dtype={args.precision}")
    elif args.precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {args.precision}")

    x_vae = preprocess(read_video(args.video_path, args.num_frames, args.sample_rate), args.resolution, args.crop_size)
    dtype = get_precision(args.precision)
    x_vae = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w

    if args.enable_time_chunk:
        video_recon = process_in_chunks(x_vae, vae, 7, 2)
    else:
        latents = vae.encode(x_vae)
        latents = latents.to(dtype)
        video_recon = vae.decode(latents)  # b t c h w

    save_fp = os.path.join(args.output_path, args.rec_path)
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
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--model_path", type=str, default="results/pretrained")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--enable_time_chunk", action="store_true")
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
        default="must_keep_origin_dtype",
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
    args = parser.parse_args()
    main(args)
