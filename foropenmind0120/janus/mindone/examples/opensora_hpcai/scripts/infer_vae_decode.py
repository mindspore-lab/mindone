import argparse
import datetime
import glob
import logging
import os
import sys

import numpy as np
import yaml
from tqdm import tqdm

import mindspore as ms
from mindspore import ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from opensora.models.vae.vae import SD_CONFIG, AutoencoderKL
from opensora.utils.model_utils import _check_cfgs_in_parser, str2bool

from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    return device_id


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"{args.output_path}/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    init_env(args)
    set_random_seed(args.seed)

    # 2. model initiate and weight loading
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    def vae_decode(x, micro_batch_size=None, scale_factor=1.0):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        if micro_batch_size is None:
            y = vae.decode(x / scale_factor)
            y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
            # (b 3 H W) -> (b H W 3)
            y = ops.transpose(y, (0, 2, 3, 1))
            y = y.asnumpy()
        else:
            bs = micro_batch_size
            y_out = []
            for i in tqdm(range(0, x.shape[0], bs)):
                x_bs = x[i : min(i + bs, x.shape[0])]
                y_bs = vae.decode(x_bs / scale_factor).asnumpy()
                y_out.append(y_bs)
            y = np.concatenate(y_out)
            y = np.clip((y + 1.0) / 2.0, a_min=0.0, a_max=1.0)
            y = np.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_from_diffusion_output(x):
        out = []
        for x_sample in x:
            # c t h w -> t c h w
            x_sample = x_sample.permute(1, 0, 2, 3)
            out.append(vae_decode(x_sample, scale_factor=args.sd_scale_factor))
        out = ops.stack(out, axis=0)

        return out

    latent_paths = sorted(glob.glob(os.path.join(args.latent_folder, "*.npy")))
    npz_format = False
    if len(latent_paths) == 0:
        latent_paths = sorted(glob.glob(os.path.join(args.latent_folder, "*.npz")))
        npz_format = True

    logger.info(f"Num samples: {len(latent_paths)}")
    for lpath in latent_paths:
        if npz_format:
            data = np.load(lpath)
            z = data["latent_mean"]
            # sample_from_distribution = 'latent_std' in data.keys()
            sample_from_distribution = False
            if sample_from_distribution:
                latent_std = data["latent_std"]
                z = z + latent_std * np.random.standard_normal(z.shape).astype(np.float32)
            z = ms.Tensor(z)
            logger.info(f"Decoding latent of shape {z.shape} from {lpath}")
            # NOTE: here we assue we directly decode from vae encoding output, not scale is applied before.
            vid = vae_decode(z, micro_batch_size=8, scale_factor=1.0)
            vid = np.expand_dims(vid, axis=0)
        else:
            z = np.load(lpath)
            z = ms.Tensor(z)
            logger.info(f"Decoding latent of shape {z.shape} from {lpath}")
            vid = vae_decode_from_diffusion_output(z)

        assert vid.shape[0] == 1
        fn = os.path.basename(lpath)[:-4]
        save_fp = f"{save_dir}/{fn}.{args.save_format}"
        save_videos(vid, save_fp, fps=args.fps)
        logger.info(f"Video saved in {save_fp}")


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
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument(
        "--latent_folder",
        type=str,
        default=["samples/denoised_latents"],
        help="path to folder containing denoised latents",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="samples",
        help="output dir to save the generated videos",
    )
    parser.add_argument(
        "--use_fp16",
        default=False,
        type=str2bool,
        help="whether use fp16.",
    )
    parser.add_argument(
        "--save_format",
        default="mp4",
        choices=["gif", "mp4"],
        type=str,
        help="video format for saving the sampling output, gif or mp4",
    )
    parser.add_argument("--fps", type=int, default=8, help="FPS in the saved video")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")

    default_args = parser.parse_args()
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = to_abspath(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    # convert to absolute path, necessary for modelarts
    args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    args.latent_folder = to_abspath(abs_path, args.latent_folder)
    args.output_path = to_abspath(abs_path, args.output_path)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
