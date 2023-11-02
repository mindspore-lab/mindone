import importlib
import random
from inspect import isfunction

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.train.amp import AMP_BLACK_LIST, AMP_WHITE_LIST, _auto_black_list, _auto_white_list


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def expand_dims_like(x, y):
    dim_diff = y.dim() - x.dim()
    if dim_diff > 0:
        for _ in range(dim_diff):
            x = x.unsqueeze(-1)
    return x


def count_params(model, verbose=False):
    total_params = sum([p.asnumpy().size for _, p in model.parameters_and_names()])  # tensor.numel()
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x):
    return np.concatenate([x, np.zeros([1], dtype=x.dtype)])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    for i in range(dims_to_append):
        x = x[..., None]
    return x


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def auto_mixed_precision(network, amp_level="O0"):
    """
    auto mixed precision function.

    Args:
        network (Cell): Definition of the network.
        amp_level (str): Supports ["O0", "O1", "O2", "O3"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O2": Cast network to float16, keep operators in black_list run in float32,
            - "O3": Cast network to float16.

    Raises:
        ValueError: If amp level is not supported.

    Examples:
        >>> from mindspore import amp, nn
        >>> network = LeNet5()
        >>> amp_level = "O1"
        >>> net = amp.auto_mixed_precision(network, amp_level)
    """

    if not isinstance(network, nn.Cell):
        raise TypeError("The network type should be Cell.")

    if amp_level == "O0":
        pass
    elif amp_level == "O1":
        return _auto_white_list(network, AMP_WHITE_LIST)
    elif amp_level == "O2":
        _auto_black_list(
            network,
            AMP_BLACK_LIST
            + [
                nn.GroupNorm,
            ],
        )
    elif amp_level == "O3":
        network.to_float(ms.float16)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))
    return network
