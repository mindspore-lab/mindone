"""
Use small-size video input to verify the correctness of the VAE.
python tests/test_vae.py --video-path test.mp4 --height 64 --width 64 --num-frames 9
"""
import argparse
import logging
import os
import sys

import torch

import mindspore as ms
from mindspore import mint

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
import numpy as np

from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

sys.path.append(".")

from hyvideo.constants import PRECISION_TO_TYPE, PRECISIONS, VAE_PATH
from hyvideo.utils.data_utils import preprocess_video, read_video
from hyvideo.utils.ms_utils import init_env
from hyvideo.vae import load_vae

logger = logging.getLogger(__name__)


def print_diff(x_ms, x_torch, name):
    abs_diff = np.abs(x_ms - x_torch).mean()
    rel_diff = (np.abs(x_ms - x_torch) / (np.abs(x_torch) + 1e-8)).mean()
    rel_diff2 = (np.abs(x_ms - x_torch) / (np.abs(x_torch) + np.abs(x_torch).mean())).mean()
    print("{} abs_diff: {:.4f}, rel_diff: {:.4f}, rel_diff2: {:.4f}".format(name, abs_diff, rel_diff, rel_diff2))


def test_vae_encoder_decoder(args, vae, vae_torch, dtype):
    x_vae = preprocess_video(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
    input_ms = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    # ms reconstruction
    sample_posterior = False
    latents = vae.encode(input_ms, sample_posterior=sample_posterior)
    encoder_output_ms = latents.to(ms.float32).asnumpy()
    latents = latents.to(dtype)
    recon = vae.decode(latents)  # b c t h w
    decoder_output_ms = recon.to(ms.float32).asnumpy()

    # torch reconstruction for fp32
    torch_dtype = torch.float32
    input_torch = torch.Tensor(x_vae).unsqueeze(0).to(torch_dtype)
    latents = vae_torch.encode(input_torch, return_dict=False)[0].mean
    encoder_output_torch = latents.detach().to(torch.float32).cpu().numpy()
    latents = torch.Tensor(encoder_output_ms).to(torch_dtype)  # force the input to decoder is the same for ms and torch
    recon = vae_torch.decode(latents, return_dict=False)[0]
    decoder_output_torch = recon.detach().to(torch.float32).cpu().numpy()

    # compare differences between torch and ms outputs
    print_diff(encoder_output_ms, encoder_output_torch, name="encoder output")
    print_diff(decoder_output_ms, decoder_output_torch, name="decoder output")


def test_vae_encoding(args, vae, vae_torch, dtype):
    x_vae = preprocess_video(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
    input_ms = ms.Tensor(x_vae, dtype).unsqueeze(0)

    # mindspore output
    x = input_ms

    if vae.use_slicing and x.shape[0] > 1:
        encoded_slices = [vae.encoder(x_slice) for x_slice in mint.split(x, 1)]
        h_ms = mint.cat(encoded_slices)
    else:
        h_ms = vae.encoder(x)

    moments_ms = vae.quant_conv(h_ms)
    posterior_mean_ms, _ = mint.split(moments_ms, [moments_ms.shape[1] // 2, moments_ms.shape[1] // 2], dim=1)
    h_ms_np = h_ms.asnumpy().astype(np.float32)
    moments_ms_np = moments_ms.asnumpy().astype(np.float32)
    posterior_mean_ms_np = posterior_mean_ms.asnumpy().astype(np.float32)

    # torch output
    torch_dtype = torch.float32
    input_torch = torch.Tensor(x_vae).unsqueeze(0).to(torch_dtype)
    x = input_torch
    if vae_torch.use_slicing and x.shape[0] > 1:
        encoded_slices = [vae_torch.encoder(x_slice) for x_slice in x.split(1)]
        h_torch = torch.cat(encoded_slices)
    else:
        h_torch = vae_torch.encoder(x)

    moments_torch = vae_torch.quant_conv(h_torch)
    posterior_mean_torch, _ = torch.chunk(moments_torch, 2, dim=1)
    h_torch_np = h_torch.detach().cpu().numpy().astype(np.float32)
    moments_torch_np = moments_torch.detach().cpu().numpy().astype(np.float32)
    posterior_mean_torch_np = posterior_mean_torch.detach().cpu().numpy().astype(np.float32)

    print_diff(h_ms_np, h_torch_np, "h")
    print_diff(moments_ms_np, moments_torch_np, "moments")
    print_diff(posterior_mean_ms_np, posterior_mean_torch_np, "posterior_mean")


def test_vae_encoder(args, vae, vae_torch, dtype):
    x_vae = preprocess_video(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
    input_ms = ms.Tensor(x_vae, dtype).unsqueeze(0)

    # mindspore output
    conv_in_out_ms = vae.encoder.conv_in(input_ms)

    # down
    down_block_out_ms = []
    sample = conv_in_out_ms
    for down_block in vae.encoder.down_blocks:
        sample = down_block(sample)
        down_block_out_ms.append(sample)

    # middle
    mid_block_out_ms = vae.encoder.mid_block(sample)

    # post-process
    conv_norm_out_ms = vae.encoder.conv_norm_out(mid_block_out_ms)
    conv_act_out_ms = vae.encoder.conv_act(conv_norm_out_ms)
    conv_out_ms = vae.encoder.conv_out(conv_act_out_ms)

    # torch output
    torch_dtype = torch.float32
    input_torch = torch.Tensor(x_vae).unsqueeze(0).to(torch_dtype)

    conv_in_out_torch = vae_torch.encoder.conv_in(input_torch)

    # down
    down_block_out_torch = []
    sample = conv_in_out_torch
    for down_block in vae_torch.encoder.down_blocks:
        sample = down_block(sample)
        down_block_out_torch.append(sample)

    # middle
    mid_block_out_torch = vae_torch.encoder.mid_block(sample)

    # post-process
    conv_norm_out_torch = vae_torch.encoder.conv_norm_out(mid_block_out_torch)
    conv_act_out_torch = vae_torch.encoder.conv_act(conv_norm_out_torch)
    conv_out_torch = vae_torch.encoder.conv_out(conv_act_out_torch)

    # compare diff
    print_diff(
        conv_in_out_ms.asnumpy().astype(np.float32),
        conv_in_out_torch.detach().cpu().numpy().astype(np.float32),
        "conv_in_out",
    )
    for i in range(len(down_block_out_ms)):
        print_diff(
            down_block_out_ms[i].asnumpy().astype(np.float32),
            down_block_out_torch[i].detach().cpu().numpy().astype(np.float32),
            f"down_block_{i}",
        )
    print_diff(
        mid_block_out_ms.asnumpy().astype(np.float32),
        mid_block_out_torch.detach().cpu().numpy().astype(np.float32),
        "mid_block",
    )
    print_diff(
        conv_norm_out_ms.asnumpy().astype(np.float32),
        conv_norm_out_torch.detach().cpu().numpy().astype(np.float32),
        "conv_norm_out",
    )
    print_diff(
        conv_act_out_ms.asnumpy().astype(np.float32),
        conv_act_out_torch.detach().cpu().numpy().astype(np.float32),
        "conv_act_out",
    )
    print_diff(
        conv_out_ms.asnumpy().astype(np.float32), conv_out_torch.detach().cpu().numpy().astype(np.float32), "conv_out"
    )


def main(args):
    init_env(
        mode=args.mode,
        device_target=args.device,
        precision_mode=args.precision_mode,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
        seed=args.seed,
    )

    set_logger(name="", output_dir=args.output_path, rank=0)

    vae, _, _, _ = load_vae(
        args.vae,
        logger=logger,
        precision=args.vae_precision,
        checkpoint=args.ms_checkpoint,
        tiling=args.vae_tiling,
    )
    dtype = PRECISION_TO_TYPE[args.vae_precision]

    # load vae in torch
    assert os.path.exists(
        "torch_hyvideo"
    ), "Please download hyvideo from https://github.com/Tencent/HunyuanVideo/tree/main/hyvideo and name it as ./torch_hyvideo"
    from torch_hyvideo.vae import load_vae as load_vae_torch

    vae_torch, _, _, _ = load_vae_torch(
        args.vae,
        "fp32",  # torch uses fp32
        logger=logger,
        device="cpu",
    )
    if args.vae_tiling:
        vae_torch.enable_tiling()
    vae_torch.eval()

    if args.input_type == "video":
        # test_vae_encoding(args, vae, vae_torch, dtype)
        # test_vae_encoder(args, vae, vae_torch, dtype)
        test_vae_encoder_decoder(args, vae, vae_torch, dtype)
    else:
        raise ValueError("Unsupported input type. Please choose from 'image', 'video', or 'folder'.")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-type",
        type=str,
        default="video",
        choices=["image", "video", "folder"],
        help="Type of input data: image, video, or folder.",
    )
    parser.add_argument(
        "--output-path", type=str, default="save_videos/", help="Path to save the reconstructed video or image path."
    )
    # Image Group
    parser.add_argument("--image-path", type=str, default="", help="Path to the input image file")

    # Video Group
    parser.add_argument(
        "--height", type=int, default=336, help="Height of the processed video frames. It applies to image size too."
    )
    parser.add_argument(
        "--width", type=int, default=336, help="Width of the processed video frames. It applies to image size too."
    )
    parser.add_argument("--video-path", type=str, default="", help="Path to the input video file.")
    parser.add_argument(
        "--rec-path",
        type=str,
        default="",
        help="Path to save the reconstructed video/image path, relative to the given output path.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument("--num-frames", type=int, default=65, help="Number of frames to sample from the video.")
    parser.add_argument("--sample-rate", type=int, default=1, help="Sampling rate for video frames.")
    parser.add_argument("--sample-fps", type=int, default=30, help="Sampling frames per second for the video.")

    # Video Folder Group
    parser.add_argument(
        "--real-video-dir", type=str, default="", help="Directory containing real videos for processing."
    )
    parser.add_argument("--generated-video_dir", type=str, default="", help="Directory to save generated videos.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument(
        "--data-file-path",
        default=None,
        help="Data file path where video paths are recorded. Supports json and csv files. "
        "If not provided, will search all videos under `real_video_dir` recursively.",
    )

    # Other Group
    parser.add_argument(
        "--vae", type=str, default="884-16c-hy", choices=list(VAE_PATH), help="Name of the VAE model to use."
    )
    parser.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model: fp16, bf16, or fp32.",
    )
    parser.add_argument("--vae-tiling", action="store_true", help="Enable tiling for the VAE model to save GPU memory.")
    parser.add_argument("--ms-checkpoint", type=str, default=None, help="Path to the MindSpore checkpoint file.")
    parser.add_argument(
        "--dynamic-start-index", action="store_true", help="Use dynamic start index for video sampling."
    )
    parser.add_argument("--expand_dim_t", default=False, type=str2bool, help="Expand dimension t for the dataset.")
    # MindSpore setting
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode.")
    parser.add_argument("--device", type=str, default="Ascend", help="Device to run the model on: Ascend or GPU.")
    parser.add_argument(
        "--max-device-memory",
        type=str,
        default=None,
        help="Maximum device memory to use, e.g., `30GB` for 910a, `59GB` for 910b.",
    )
    parser.add_argument("--use-parallel", default=False, type=str2bool, help="Use parallel processing.")
    parser.add_argument(
        "--parallel-mode", default="data", type=str, choices=["data", "optim"], help="Parallel mode: data or optim."
    )
    parser.add_argument("--jit-level", default="O0", help="Set JIT level: O0: KBK, O1: DVM, O2: GE.")
    parser.add_argument(
        "--jit-syntax-level", default="strict", choices=["strict", "lax"], help="Set JIT syntax level: strict or lax."
    )
    parser.add_argument("--seed", type=int, default=4, help="Random seed for inference.")
    parser.add_argument(
        "--precision-mode", default=None, type=str, help="Set precision mode for Ascend configurations."
    )
    parser.add_argument(
        "--vae-keep-gn-fp32",
        default=False,
        type=str2bool,
        help="Keep GroupNorm in fp32. Defaults to False in inference, better to set to True when training VAE.",
    )
    parser.add_argument(
        "--dataset-name", default="video", type=str, choices=["image", "video"], help="Dataset name: image or video."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
