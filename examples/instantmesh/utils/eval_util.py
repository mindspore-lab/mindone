import logging
import math
import os
import sys
from typing import List, Optional, Tuple, Union

from PIL import Image

import mindspore as ms
from mindspore import mint, ops

from mindone.utils.seed import set_random_seed

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../../..")))  # for loading mindone
logger = logging.getLogger("")


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def init_inference_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    device_id: int = 7,
    jit_level: str = "O0",
    debug: bool = False,
):
    """
    Init the MS env for inference.
    """
    set_random_seed(seed)
    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    device_num = 1
    rank_id = 0
    ms.set_context(
        mode=mode,
        device_target=device_target,
        pynative_synchronize=debug,
        jit_config={"jit_level": jit_level},
        device_id=device_id,
    )
    return rank_id, device_num


def make_grid_ms(
    tensor: ms.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> ms.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = mint.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = mint.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = mint.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        # tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img = mint.clamp(img, min=low, max=high)
            img = mint.sub(img, low)
            img = mint.div(img, max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, ms.Tensor):
        raise TypeError("tensor should be of type ms Tensor")

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    # grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    grid = mint.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            # if k >= nmaps:
            #     break
            grid[:, y * height + padding : y * height + height, x * width + padding : x * width + width].copy_(
                tensor[k]
            )

            k = k + 1
    return grid


def save_image_ms(
    tensor: Union[ms.Tensor, List[ms.Tensor]],
    fp: str,
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid_ms(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to(ms.uint8).numpy()

    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def get_params_ms(img: ms.Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    _, h, w = img.shape
    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    if w == tw and h == th:
        return 0, 0, h, w

    i = ops.randint(0, h - th + 1, size=(1,)).item()
    j = ops.randint(0, w - tw + 1, size=(1,)).item()
    return i, j, th, tw
