
import math

import einops
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor

from mindspore.common.initializer import (
    Constant,
    Normal,
    One,
    TruncatedNormal,
    XavierNormal,
    XavierUniform,
    Zero,
    initializer,
)

class Normalize(nn.Cell):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def construct(self, x):
        return normalize(x, dim=self.dim, p=2)

def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """

    denom = input.norm(p, dim, keepdim=True).clip(min=eps).broadcast_to(input.shape)

    return input / denom


class LearnableLogitScaling(nn.Cell):
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = ops.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = Parameter(log_logit_scale)
        else:
            self.log_logit_scale = Parameter(log_logit_scale, requires_grad=False)

    def construct(self, x):
        return ops.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self):
        st = f"logit_scale_init={self.logit_scale_init},learnable={self.learnable}, max_logit_scale={self.max_logit_scale}"
        return st


class EinOpsRearrange(nn.Cell):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def construct(self, x):
        assert isinstance(x, ms.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


class VerboseNNModule(nn.Cell):
    """
    Wrapper around nn.Cell that prints registered buffers and parameter names.
    """

    @staticmethod
    def get_readable_tensor_repr(name: str, tensor: ms.Tensor) -> str:
        st = (
            "("
            + name
            + "): "
            + "tensor("
            + str(tuple(tensor[1].shape))
            + ", requires_grad="
            + str(tensor[1].requires_grad)
            + ")\n"
        )
        return st

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr += self.get_readable_tensor_repr(name, p)

        for p in self.named_buffers():
            name = p[0].split(".")[0]
            string_repr += self.get_readable_tensor_repr(name, p)

        return string_repr


def cast_if_src_dtype(
    tensor: ms.Tensor, src_dtype: ms.dtype, tgt_dtype: ms.dtype
):
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.to(dtype=tgt_dtype)
        updated = True
    return tensor, updated


class QuickGELU(nn.Cell):
    # From https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L166
    def construct(self, x: ms.Tensor):
        return x * ops.sigmoid(1.702 * x)


class SelectElement(nn.Cell):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def construct(self, x):
        assert x.ndim >= 3
        return x[:, self.index, ...]


class SelectEOSAndProject(nn.Cell):
    """
    Text Pooling used in OpenCLIP
    """

    def __init__(self, proj: nn.Cell) -> None:
        super().__init__()
        self.proj = proj

    def construct(self, x, seq_len):
        assert x.ndim == 3
        # x is of shape B x L x D
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[ops.arange(x.shape[0]), seq_len]
        x = self.proj(x)
        return x

def trunc_normal_(tensor: Parameter, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> None:
    tensor.set_data(initializer(TruncatedNormal(std, mean, a, b), tensor.shape, tensor.dtype))

def constant_(tensor: Parameter, val: float) -> None:
    tensor.set_data(initializer(Constant(val), tensor.shape, tensor.dtype))

def normal_(tensor: Parameter, mean: float = 0.0, std: float = 1.0) -> None:
    tensor.set_data(initializer(Normal(std, mean), tensor.shape, tensor.dtype))

def zeros_(tensor: Parameter) -> None:
    tensor.set_data(initializer(Zero(), tensor.shape, tensor.dtype))

class DropPath(nn.Cell):

    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(p=drop_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor