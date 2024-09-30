import math
import numpy as np
from inspect import isfunction
import mindspore as ms
from mindspore import nn, ops, mint


class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 5:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def construct(self, x: ms.Tensor):
        orig_type = x.dtype
        ret = super().construct(ops.cast(x, ms.float32))
        return ops.cast(ret, orig_type)


def gather_data(data, return_np=True):
    """gather data from multiple processes to one list"""
    data_list = [mint.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(data_list, data)  # gather not supported with NCCL
    if return_np:
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list


def autocast(f):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=True,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, repeat=False):
    if not repeat:
        return ms.ops.StandardNormal()(shape)
    else:
        raise ValueError("The repeat method os nor supported.")


def default(val, d):
    if exists(val):
        return val
    if isinstance(d, (ms.Tensor, int, float)):
        return d
    return d()


def exists(val):
    return val is not None


def identity(*args, **kwargs):
    return nn.Identity()


def uniq(arr):
    return {el: True for el in arr}.keys()


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def ismap(x):
    if not isinstance(x, ms.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, ms.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def max_neg_value(t):
    return -np.finfo(t.dtype).max


def shape_to_str(x):
    shape_str = "x".join([str(x) for x in x.shape])
    return shape_str


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """

    return func(*inputs)

    # if flag:
    #     ckpt = ms.utils.checkpoint.checkpoint
    #     return ckpt(func, *inputs)
    # else:
    #     return func(*inputs)