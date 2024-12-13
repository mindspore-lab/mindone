#!/usr/bin/env python
import argparse
import glob
import logging
import os
from typing import Dict

from fileio import load_safetensors
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor

logger = logging.getLogger(__name__)


def load(root_path: str) -> Dict[str, Tensor]:
    # TODO: this method may cause OOM on computer with low memory
    # use a better solution later
    pattern = os.path.join(root_path, "*.safetensors")
    filelist = sorted(glob.glob(pattern))

    filenames = [os.path.basename(x) for x in filelist]
    logger.info(f"Files need to be converted: `{filenames}`")

    params: Dict[str, Tensor] = dict()
    for x in tqdm(filelist, desc="Loading the safetensors"):
        params_chunk = load_safetensors(x)
        if params.keys().isdisjoint(params_chunk.keys()):
            params.update(params_chunk)
        else:
            same_keys = set(params.keys()).intersection(params_chunk.keys())
            raise RuntimeError(f"Duplicated keys found: `{same_keys}`.")
    return params


def convert(params: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # compatibility between MS and PyTorch naming and formating
    return params


def save(ckpt: Dict[str, Tensor], output: str) -> None:
    output = os.path.abspath(output)
    logger.info(f"Saving to {output}...")
    ms.save_checkpoint(ckpt, output)
    logger.info(f"Saving to {output}...Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert LLaVa checkpoints into Mindspore Format")
    parser.add_argument("src", help="Directory storing the safetensors")
    parser.add_argument(
        "-o", "--output", default="models/llava_1_6.ckpt", help="Name of the output Mindspore checkpoint"
    )

    args = parser.parse_args()

    params = load(args.src)
    params = convert(params)
    save(params, args.output)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    main()
