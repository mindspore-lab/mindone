import os
from inspect import isfunction
from typing import List, Optional, Tuple, Union

import mindspore as ms
from mindspore import ops


def extract_into_tensor(a: ms.Tensor, t: ms.Tensor, x_shape: Union[Tuple, List]):
    """
    Extract elements in tensor `a` by indices `t` and reshape them to `x_shape`.
    Frequently used in extracting sqrt prod alphas for diffusion forward sampling in batch training.

    Args:
        a: ms.Tensor of float, e.g. sqrt_alphas_cumprod, in shape (timesteps, )
        t: indices for retrieving values from `a`, e.g. multiple sqrt_alpha_cumprod_t, in shape (batch_size, )
        x_shape: a tuple or list to indicate the shape of input sample (to add noise), e.g. (b, c, h, w)
    Return:
        extracted alphas in shape (b, 1, 1, 1)
    """
    b = t.shape[0]
    out = ops.GatherD()(a, -1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_abspath(root_dir: str, tmp_path: Optional[str]):
    if not tmp_path or tmp_path.startswith("/"):
        return tmp_path
    return os.path.join(root_dir, tmp_path)
