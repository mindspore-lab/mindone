import argparse
import logging
import os
import sys

import torch

import mindspore as ms

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
import numpy as np

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

sys.path.append(".")

from hyvideo.constants import PRECISION_TO_TYPE, PRECISIONS, VAE_PATH
from hyvideo.utils.data_utils import preprocess_video, read_video
from hyvideo.utils.ms_utils import init_env
from hyvideo.vae import load_vae
from hyvideo.vae.unet_causal_3d_blocks import GroupNorm, MSInterpolate, MSPad

logger = logging.getLogger(__name__)


def print_diff(x_ms, x_torch):
    abs_diff = np.abs(x_ms - x_torch).mean()
    rel_diff = (np.abs(x_ms - x_torch) / (np.abs(x_torch) + 1e-8)).mean()
    rel_diff2 = (np.abs(x_ms - x_torch) / (np.abs(x_torch) + np.abs(x_torch).mean())).mean()
    print("abs_diff: {:.4f}, rel_diff: {:.4f}, rel_diff2: {:.4f}".format(abs_diff, rel_diff, rel_diff2))


def test_video(args, vae, vae_torch, dtype):
    x_vae = preprocess_video(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
    input_ms = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    # ms reconstruction
    sample_posterior = False
    latents = vae.encode(input_ms, sample_posterior=sample_posterior)
    encoder_output_ms = latents.to(ms.float32).asnumpy()
    latents = latents.to(dtype)
    recon = vae.decode(latents)  # b c t h w
    decoder_output_ms = recon.to(ms.float32).asnumpy()

    # torch reconstruction
    if dtype == ms.float32:
        torch_dtype = torch.float32
    elif dtype == ms.float16:
        torch_dtype = torch.float16
    elif dtype == ms.bfloat16:
        torch_dtype = torch.bfloat16
    else:
        raise ValueError

    vae_torch.eval()
    input_torch = torch.Tensor(x_vae).unsqueeze(0).to(torch_dtype)
    latents = vae_torch.encode(input_torch, return_dict=False)[0].mean
    encoder_output_torch = latents.detach().to(torch.float32).cpu().numpy()
    latents = latents.to(torch_dtype)
    recon = vae_torch.decode(latents)
    decoder_output_torch = recon.detach().to(torch.float32).cpu().numpy()

    # compare differences between torch and ms outputs
    print("Compare encoder output")
    print_diff(encoder_output_ms, encoder_output_torch)
    print("Compare decoder output")
    print_diff(decoder_output_ms, decoder_output_torch)


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

    if args.ms_checkpoint is not None and os.path.exists(args.ms_checkpoint):
        logger.info(f"Run inference with MindSpore checkpoint {args.ms_checkpoint}")
        state_dict = ms.load_checkpoint(args.ms_checkpoint)
        state_dict = dict(
            [k.replace("autoencoder.", "") if k.startswith("autoencoder.") else k, v] for k, v in state_dict.items()
        )
    else:
        state_dict = None

    vae, _, s_ratio, t_ratio = load_vae(
        args.vae,
        logger=logger,
        state_dict=state_dict,
    )

    if args.vae_tiling:
        vae.enable_tiling()

    if args.vae_precision in ["fp16", "bf16"]:
        amp_level = "O2"
        dtype = PRECISION_TO_TYPE[args.vae_precision]
        if dtype == ms.float16:
            custom_fp32_cells = [GroupNorm] if args.vae_keep_gn_fp32 else []
        else:
            custom_fp32_cells = [MSPad, MSInterpolate]

        vae = auto_mixed_precision(vae, amp_level, dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(
            f"Set mixed precision to {amp_level} with dtype={args.vae_precision}, custom fp32_cells {custom_fp32_cells}"
        )
    elif args.vae_precision == "fp32":
        dtype = PRECISION_TO_TYPE[args.vae_precision]
    else:
        raise ValueError(f"Unsupported precision {args.vae_precision}")

    # load vae in torch
    # download https://github.com/Tencent/HunyuanVideo/tree/main/hyvideo/vae and place it under ./torch_vae
    assert os.path.exists(
        "torch_vae"
    ), "Please download torch_vae from https://github.com/Tencent/HunyuanVideo/tree/main/hyvideo/vae and place it under ./torch_vae"
    from torch_vae import load_vae as load_vae_torch

    vae_torch, _, s_ratio, t_ratio = load_vae_torch(
        args.vae,
        args.vae_precision,
        logger=logger,
        device="cpu",
    )
    if args.vae_tiling:
        vae_torch.enable_tiling()

    if args.input_type == "video":
        test_video(args, vae, vae_torch, dtype)
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
