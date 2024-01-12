import importlib
import random
from inspect import isfunction

import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
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


clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor]: clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(
            grad, F.cast(F.tuple_to_array((-clip_value,)), dt), F.cast(F.tuple_to_array((clip_value,)), dt)
        )
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


apply_global_norm = C.MultitypeFuncGraph("_apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, x):
    x_dtype = F.dtype(x)
    x = x * clip_norm / global_norm
    x = F.cast(x, x_dtype)
    return x


class L2Norm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.l2_norm_1 = ops.LpNorm((0,))
        self.l2_norm_2 = ops.LpNorm((0, 1))
        self.l2_norm_3 = ops.LpNorm((0, 1, 2))
        self.l2_norm_4 = ops.LpNorm((0, 1, 2, 3))

    def construct(self, x):
        if x.ndim == 1:
            norm = self.l2_norm_1(x)
        elif x.ndim == 2:
            norm = self.l2_norm_2(x)
        elif x.ndim == 3:
            norm = self.l2_norm_3(x)
        else:
            norm = self.l2_norm_4(x)
        return norm


class _ClipByGlobalNormFix(nn.Cell):
    def __init__(self, clip_norm=1.0):
        super().__init__()
        self.clip_norm = Tensor([clip_norm], ms.float32)
        self.hyper_map = ops.HyperMap()
        self.greater_equal = ops.GreaterEqual()
        self.l2norm = L2Norm()

    def construct(self, x):
        norms = self.hyper_map(self.l2norm, x)
        norms_square = self.hyper_map(ops.square, norms)
        global_norm = ops.sqrt(ops.addn(norms_square)).astype(ms.float32)

        cond = self.greater_equal(global_norm, self.clip_norm)
        global_norm = F.select(cond, global_norm, self.clip_norm)
        clip_x = self.hyper_map(F.partial(apply_global_norm, self.clip_norm, global_norm), x)
        return clip_x


hyper_map_op = ops.HyperMap()


def clip_grad_(x, clip_norm=1.0):
    clip_value = hyper_map_op(F.partial(clip_grad, 1, clip_norm), x)
    return clip_value


def clip_grad_global_(x, clip_norm=1.0):
    clip_value = _ClipByGlobalNormFix(clip_norm)(x)
    return clip_value


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
        try:
            _auto_black_list(
                network,
                AMP_BLACK_LIST
                + [
                    nn.GroupNorm,
                ],
                ms.float16,
            )
        except Exception:
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
