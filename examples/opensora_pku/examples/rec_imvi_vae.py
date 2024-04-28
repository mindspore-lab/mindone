"""
Run causal vae reconstruction on a given video.
Usage example:
python examples/rec_imvi_vae.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0 \
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
from opensora.models.ae import getae_wrapper
from opensora.utils.dataset_utils import create_video_transforms
from PIL import Image

import mindspore as ms
from mindspore import nn, ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
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
    video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    return video_data


def preprocess(video_data, sample_size=128):
    num_frames = video_data.shape[1]
    video_transform = create_video_transforms(
        sample_size, sample_size, num_frames=num_frames, backend="al", disable_flip=True
    )

    inputs = {"image": video_data[:, 0]}
    for i in range(num_frames - 1):
        inputs[f"image{i}"] = video_data[:, i + 1]

    video_outputs = video_transform(**inputs)
    video_outputs = (video_outputs / 255.0) * 2 - 1.0
    video_outputs = np.stack(list(video_outputs.values()), axis=0)
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
    num_frames = video_data.size(2)
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


def main(args):
    init_env(args)
    set_logger(name="", output_dir=args.output_path, rank=0)

    kwarg = {}
    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir="cache_dir", **kwarg)
    # if args.enable_tiling:
    #     vae.vae.enable_tiling()
    #     vae.vae.tile_overlap_factor = args.tile_overlap_factor

    vae.set_train(False)
    for param in vae.get_parameters():
        param.requires_grad = False
    if args.dtype != "fp32":
        amp_level = "O2"
        dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        vae = auto_mixed_precision(vae, amp_level, dtype)
        logger.info(f"Set mixed precision to O2 with dtype={args.dtype}")
    else:
        amp_level = "O0"

    x_vae = preprocess(read_video(args.video_path, args.num_frames, args.sample_rate), args.sample_size)
    x_vae = x_vae  # b c t h w
    if args.enable_time_chunk:
        video_recon = process_in_chunks(x_vae, vae, 7, 2)
    else:
        latents = vae.encode(x_vae)
        latents = latents.to(ms.float16)
        video_recon = vae.decode(latents)  # b t c h w

    if video_recon.shape[2] == 1:
        x = video_recon[0, 0, :, :, :]
        x = x.squeeze().asnumpy()
        x = np.clip(x, -1, 1)
        x = (x + 1) / 2
        x = (255 * x).astype(np.uint8)
        x = x.transpose(1, 2, 0)
        image = Image.fromarray(x)
        image.save(args.rec_path.replace("mp4", "jpg"))
    else:
        save_video_data = video_recon[0].transpose(0, 2, 3, 4, 1)  # (b c t h w) -> (b t h w c)
        save_videos(save_video_data, args.rec_path, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--rec_path", type=str, default="")
    parser.add_argument("--ae", type=str, default="")
    parser.add_argument("--model_path", type=str, default="results/pretrained")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--sample_rate", type=int, default=1)
    # parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    # parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument("--enable_time_chunk", action="store_true")
    # ms related
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="mixed precision type, if fp32, all layer precision is float32 (amp_level=O0),  \
                if bf16 or fp16, amp_level==O2, part of layers will compute in bf16 or fp16 such as matmul, dense, conv.",
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument(
        "--precision_mode",
        default="must_keep_origin_dtype",
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    args = parser.parse_args()
    main(args)
