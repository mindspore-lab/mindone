#!/usr/bin/env python
import argparse
import csv
import logging
import os
import sys
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms import Compose, vision

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from pixart.dataset.constant import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from pixart.dataset.utils import classify_height_width_bin
from pixart.modules.vae import SD_CONFIG, AutoencoderKL
from pixart.utils import str2bool

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


class ImageDataset:
    def __init__(
        self, csv_path: str, image_dir: str, image_size: int, path_column: str = "dir", multi_scale: bool = False
    ) -> None:
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.image_dir = image_dir
        self.path_column = path_column
        self.multi_scale = multi_scale

        if not self.multi_scale:
            self.ratio = None
            self.transform = self.create_transform(image_size=image_size)
        else:
            if image_size == 1024:
                self.ratio = ASPECT_RATIO_1024_BIN
            elif image_size == 512:
                self.ratio = ASPECT_RATIO_512_BIN
            elif image_size == 256:
                self.ratio = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("`image_size` must be 256, 512 or 1024 when `multi_scale=True`")
            self.transform = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[str, np.ndarray]:
        row = self.dataset[idx]
        path = row[self.path_column]
        full_path = os.path.join(self.image_dir, path)
        image = Image.open(full_path).convert("RGB")

        if not self.multi_scale:
            image = self.transform(image)[0]
        else:
            width, height = image.size
            height, width = classify_height_width_bin(height, width, self.ratio)
            transform = self.create_multi_scale_transform((height, width))
            image = transform(image)[0]

        return path, image

    @staticmethod
    def create_transform(image_size: int) -> Compose:
        return Compose(
            [
                vision.Resize(image_size, interpolation=vision.Inter.BICUBIC),
                vision.CenterCrop(image_size),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )

    @staticmethod
    def create_multi_scale_transform(image_size: Tuple[int, int]) -> Compose:
        return Compose(
            [
                vision.Resize(image_size, interpolation=vision.Inter.BICUBIC),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )


def init_env(args) -> None:
    set_random_seed(args.seed)
    ms.set_context(mode=args.mode, device_target=args.device_target, jit_config=dict(jit_level="O2"))


def main(args):
    set_logger(output_dir="logs/infer_vae")

    # init env
    args = parse_args()
    init_env(args)

    # build dataloader
    batch_size = 1 if args.multi_scale else args.batch_size
    dataset = ImageDataset(
        args.csv_path, args.image_dir, args.image_size, path_column=args.path_column, multi_scale=args.multi_scale
    )
    dataset = GeneratorDataset(dataset, column_names=["path", "image"], shuffle=False)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # model initiate and weight loading
    network = AutoencoderKL(SD_CONFIG, 4, ckpt_path=args.vae_checkpoint)
    network.set_train(False)
    for param in network.trainable_params():
        param.requires_grad = False

    if args.dtype == "fp16":
        model_dtype = ms.float16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    elif args.dtype == "bf16":
        model_dtype = ms.bfloat16
        network = auto_mixed_precision(network, amp_level="O2", dtype=model_dtype)
    else:
        model_dtype = ms.float32

    if args.multi_scale:
        network.set_inputs(Tensor(shape=(batch_size, 3, None, None), dtype=ms.float32))

    if args.output_path is None:
        output_dir = os.path.dirname(args.csv_path)
    else:
        output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    ds_iter = dataset.create_tuple_iterator(num_epochs=1)
    for paths, images in tqdm(ds_iter, total=len(dataset)):
        paths = paths.asnumpy().tolist()
        latent_mean, latent_std = network.encode_with_moments_output(images)
        latent_mean, latent_std = latent_mean.asnumpy(), latent_std.asnumpy()
        assert latent_mean.shape[0] == len(paths)

        for i in range(latent_mean.shape[0]):
            filename = os.path.splitext(paths[i])[0] + ".npz"
            filepath = os.path.join(output_dir, filename)
            np.savez(filepath, latent_mean=latent_mean[i], latent_std=latent_std[i])
    logger.info(f"Done. VAE Latent saved in {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract VAE latent from a csv file listed path of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv_path", required=True, help="path to csv annotation file.")
    parser.add_argument("--image_dir", required=True, help="image folder storing image files.")
    parser.add_argument("--path_column", default="dir", help="column name of image path in csv file.")
    parser.add_argument("--image_size", default=512, type=int, help="image size for VAE latent extraction.")
    parser.add_argument(
        "--multi_scale", default=False, type=str2bool, help="if it is true, then the VAE is extracted in multi-scale."
    )
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument(
        "--output_path",
        help="output dir to save the VAE latents, if None, will treat the parent dir of csv_path as output dir.",
    )
    parser.add_argument(
        "--vae_checkpoint",
        default="models/sd-vae-ft-ema.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )

    parser.add_argument("--device_target", default="Ascend", choices=["CPU", "GPU", "Ascend"], help="Device target")
    parser.add_argument("--mode", default=0, type=int, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1)")
    parser.add_argument("--seed", default=4, type=int, help="Inference seed")
    parser.add_argument(
        "--dtype", default="fp32", choices=["bf16", "fp16", "fp32"], help="what data type to use for VAE."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
