#!/usr/bin/env python
"""
FiT preprocessing (for training) pipeline
"""
import argparse
import json
import logging
import os
import sys

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

import numpy as np
import yaml
from data.dataset import create_dataloader_video_preprocessing
from modules.autoencoder import SD_CONFIG, AutoencoderKL
from tqdm import tqdm
from utils.model_utils import check_cfgs_in_parser, str2bool

import mindspore as ms
import mindspore.ops as ops

from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode, device_target=args.device_target)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/preprocess/video-256x256.yaml",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="image size",
    )
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--outdir", default="./latent", help="Path of the output dir")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    parser.add_argument("--dataset_mode", default="video", help="data format")
    parser.add_argument("--random_crop", default=False, type=str2bool, help="perform random crop")
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    init_env(args)

    # 2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,  # disable amp for vae
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    # 3. build dataloader
    data_config = dict(
        data_folder=args.data_path,
        sample_size=args.image_size,
        patch_size=args.patch_size,
        num_parallel_workers=args.num_parallel_workers,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        shuffle=False,
        dataset_mode=args.dataset_mode,
        random_crop=args.random_crop,
    )
    dataloader = create_dataloader_video_preprocessing(data_config)

    # 4. run inference
    records = list()
    for video, path in tqdm(dataloader.create_tuple_iterator(num_epochs=1), total=len(dataloader)):
        path = path.asnumpy().item()
        outdir = os.path.abspath(os.path.join(save_dir, os.path.basename(os.path.dirname(path))))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        dest = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".npy")
        if os.path.isfile(dest):
            continue

        n, t, c, h, w = video.shape
        video = ops.reshape(video, (-1, c, h, w))
        latent = vae.encode_with_moments_output(video)
        _, c_l, h_l, w_l = latent.shape
        latent = ops.reshape(latent, (t, c_l, h_l, w_l)).numpy().astype(np.float16)
        np.save(dest, latent)
        records.append(dict(img=path, latent=dest))

    out_json = os.path.join(save_dir, "path.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=4)
