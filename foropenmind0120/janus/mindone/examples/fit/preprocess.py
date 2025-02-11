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
from data.imagenet_dataset import create_dataloader_imagenet_preprocessing
from modules.autoencoder import SD_CONFIG, AutoencoderKL
from tqdm import tqdm
from utils.model_utils import check_cfgs_in_parser, str2bool

import mindspore as ms
import mindspore.nn as nn

from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


class DynNet(nn.Cell):
    def __init__(self, vae):
        super().__init__(auto_prefix=False)
        self.vae = vae

    def construct(self, x):
        return self.vae.encode_with_moments_output(x)


def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )

    if args.mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": args.jit_level})

    return device_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/preprocess/sd-vae-ft-mse-256x256.yaml",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="path to source torch checkpoint, which ends with .pt",
    )
    parser.add_argument("--outdir", default="./latent", help="Path of the output dir")
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--imagenet_format", type=str2bool, default=True, help="Training with ImageNet dataset format")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--num_parallel_workers", default=4, type=int, help="num workers for data loading")

    parser.add_argument(
        "--jit_level",
        default="O1",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )

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
    net = DynNet(vae)

    net = net.set_train(False)
    for param in net.get_parameters():  # freeze vae
        param.requires_grad = False

    if args.mode == ms.GRAPH_MODE:
        net.set_inputs(ms.Tensor(shape=[1, 3, None, None], dtype=ms.float32))

    # 3. build dataloader
    if args.imagenet_format:
        data_config = dict(
            data_folder=args.data_path,
            sample_size=args.image_size,
            patch_size=args.patch_size,
            num_parallel_workers=args.num_parallel_workers,
            shuffle=False,
        )
        dataloader = create_dataloader_imagenet_preprocessing(data_config)
    else:
        raise NotImplementedError("Currently only ImageNet format is supported.")

    # 4. run inference
    records = list()
    for img, path in tqdm(dataloader.create_tuple_iterator(num_epochs=1), total=len(dataloader)):
        path = path.asnumpy().item()
        outdir = os.path.abspath(os.path.join(save_dir, os.path.basename(os.path.dirname(path))))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        dest = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".npy")
        if os.path.isfile(dest):
            continue

        latent = net(img).numpy().astype(np.float16)[0]
        np.save(dest, latent)
        records.append(dict(img=path, latent=dest))

    out_json = os.path.join(save_dir, "path.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=4)
