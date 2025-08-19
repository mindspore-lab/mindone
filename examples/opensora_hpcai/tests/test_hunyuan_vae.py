"""
Use small-size video input to verify the correctness of the Hunyuan VAE.
Please run pytorch hunyuan_vae with the same settings as the mindspore one.

python tests/test_hunyuan_vae.py --video_path test.mp4 --output_pt_path output_pt.npz
"""
import argparse
import os
import sys

import numpy as np
import yaml

import mindspore as ms

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
hunyuanvideo_path = os.path.abspath("../hunyuanvideo")
sys.path.insert(0, hunyuanvideo_path)

from hyvideo.constants import PRECISION_TO_TYPE, PRECISIONS
from hyvideo.utils.data_utils import preprocess_video, read_video
from hyvideo.utils.ms_utils import init_env
from opensora.models.hunyuan_vae.autoencoder_kl_causal_3d import CausalVAE3D_HUNYUAN


def print_diff(x_ms, x_torch, name):
    abs_diff = np.abs(x_ms - x_torch).mean()
    rel_diff = (np.abs(x_ms - x_torch) / (np.abs(x_torch) + 1e-8)).mean()
    rel_diff2 = (np.abs(x_ms - x_torch) / (np.abs(x_torch) + np.abs(x_torch).mean())).mean()
    print("{} abs_diff: {:.4f}, rel_diff: {:.4f}, rel_diff2: {:.4f}".format(name, abs_diff, rel_diff, rel_diff2))


def test_vae_encoder_decoder(args, vae, dtype):
    x_vae = preprocess_video(read_video(args.video_path, args.num_frames, args.sample_rate), args.height, args.width)
    input_ms = ms.Tensor(x_vae, dtype).unsqueeze(0)  # b c t h w
    print(f"vae encoder input_ms shape: {input_ms.shape}, dtype: {input_ms.dtype}")

    # reconstruction
    sample_posterior = False
    latents = vae.encode(input_ms, sample_posterior=sample_posterior)
    encoder_output_ms = latents.to(ms.float32).asnumpy()
    latents = latents.to(dtype)
    recon = vae.decode(latents)  # b c t h w
    decoder_output_ms = recon.to(ms.float32).asnumpy()

    return encoder_output_ms, decoder_output_ms


def main(args):
    init_env(
        mode=args.mode,
        device_target=args.device,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
        seed=args.seed,
    )

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ae_config = config["ae"]
    ae_config["from_pretrained"] = args.from_pretrained
    ae_config["dtype"] = args.vae_precision
    vae = CausalVAE3D_HUNYUAN(**ae_config).set_train(False)
    dtype = PRECISION_TO_TYPE[args.vae_precision]

    encoder_output_ms, decoder_output_ms = test_vae_encoder_decoder(args, vae, dtype)

    # Load torch output, compare differences between torch and ms
    output_pt = np.load(args.output_pt_path)
    encoder_output_pt = output_pt["encoder_output"]
    decoder_output_pt = output_pt["decoder_output"]
    assert encoder_output_ms.shape == encoder_output_pt.shape
    assert encoder_output_ms.dtype == encoder_output_pt.dtype == np.float32
    assert decoder_output_ms.shape == decoder_output_pt.shape
    assert decoder_output_ms.dtype == decoder_output_pt.dtype == np.float32

    print_diff(encoder_output_ms, encoder_output_pt, name="encoder output")
    print_diff(decoder_output_ms, decoder_output_pt, name="decoder output")
    print("\n")
    print("encoder_output_ms mean and std:", encoder_output_ms.mean(), encoder_output_ms.std())
    print("encoder_output_pt mean and std:", encoder_output_pt.mean(), encoder_output_pt.std())
    print("\n")
    print("decoder_output_ms mean and std:", decoder_output_ms.mean(), decoder_output_ms.std())
    print("decoder_output_pt mean and std:", decoder_output_pt.mean(), decoder_output_pt.std())


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="configs/opensora-v2-0/inference/256px.yaml",
        help="Path to load a config yaml file.",
    )
    parser.add_argument(
        "--from_pretrained", default="ckpts/hunyuan_vae.safetensors", type=str, help="Path to the vae ckpt."
    )

    parser.add_argument("--output_pt_path", type=str, required=True, help="Path to the torch output, a npz file.")
    parser.add_argument(
        "--height", type=int, default=64, help="Height of the processed video frames. It applies to image size too."
    )
    parser.add_argument(
        "--width", type=int, default=64, help="Width of the processed video frames. It applies to image size too."
    )
    parser.add_argument("--video_path", type=str, default="", help="Path to the input video file.")
    parser.add_argument("--num_frames", type=int, default=9, help="Number of frames to sample from the video.")
    parser.add_argument("--sample_rate", type=int, default=1, help="Sampling rate for video frames.")

    parser.add_argument(
        "--vae_precision",
        type=str,
        default="fp32",
        choices=PRECISIONS,
        help="Precision mode for the VAE model: fp16, bf16, or fp32.",
    )

    # MindSpore setting
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode.")
    parser.add_argument("--device", type=str, default="Ascend", help="Device to run the model on: Ascend or GPU.")
    parser.add_argument("--jit_level", default="O0", help="Set JIT level: O0: KBK, O1: DVM, O2: GE.")
    parser.add_argument(
        "--jit_syntax_level", default="strict", choices=["strict", "lax"], help="Set JIT syntax level: strict or lax."
    )
    parser.add_argument("--seed", type=int, default=4, help="Random seed for inference.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
