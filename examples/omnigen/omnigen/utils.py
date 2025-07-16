# Adapted from https://github.com/VectorSpaceLab/OmniGen/blob/main/OmniGen/utils.py

import logging
from typing import Dict, Union

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import nn


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def crop_arr(pil_image, max_image_size):
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1

    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1

    # arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]
    return Image.fromarray(arr[crop_y1 : arr.shape[0] - crop_y2, crop_x1 : arr.shape[1] - crop_x2])


def load_ckpt_params(model: nn.Cell, ckpt: Union[str, Dict]) -> nn.Cell:
    if isinstance(ckpt, str):
        print(f"Loading {ckpt} params into network...")
        param_dict = ms.load_checkpoint(ckpt)
    else:
        param_dict = ckpt

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict, strict_load=True)
    if not (len(param_not_load) == len(ckpt_not_load) == 0):
        print(
            "Exist ckpt params not loaded: {} (total: {}), or net params not loaded: {} (total: {})".format(
                ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
            )
        )
    return model
