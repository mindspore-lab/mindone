import math
import os
from typing import List, Union

import mindspore as ms
from mindspore import mint


def make_grid(
    tensor: Union[ms.Tensor, List[ms.Tensor]], nrow: int = 8, padding: int = 2, pad_value: int = 0, **kwargs
) -> ms.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    """
    if not (ms.is_tensor(tensor) or (isinstance(tensor, list) and all(ms.is_tensor(t) for t in tensor))):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = mint.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = mint.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = mint.cat((tensor, tensor, tensor), 1)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = mint.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value, dtype=tensor.dtype)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def load_from_checkpoint(model, ckpt_fp, remove_prefix=["var."]):
    assert os.path.exists(ckpt_fp), f"checkpoint {ckpt_fp} NOT found"
    print(f"Loading ckpt {ckpt_fp} into network")
    param_dict = ms.load_checkpoint(ckpt_fp)
    keys = list(param_dict.keys())
    for pname in keys:
        for pf in remove_prefix:
            if pname.startswith(pf):
                param_dict[pname.replace(pf, "")] = param_dict.pop(pname)

    if (
        "quantize.ema_vocab_hit_SV" in param_dict
        and param_dict["quantize.ema_vocab_hit_SV"].shape[0] != model.quantize.ema_vocab_hit_SV.shape[0]
    ):
        param_dict["quantize.ema_vocab_hit_SV"] = model.quantize.ema_vocab_hit_SV
    m, u = ms.load_param_into_net(model, param_dict)
    print("net param not load: ", m, len(m))
    print("ckpt param not load: ", u, len(u))


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True
