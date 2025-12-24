# -*- coding: utf-8 -*-
"""Custom MindSpore Operators Suite

This module encapsulates custom implementations for a curated set of operators that are either unsupported or
introduced post specific MindSpore version. Recognizing the evolving nature of the framework, this suite ensures
compatibility across different MindSpore versions, particularly catering to scenarios where native support is
lacking across all versions, and require manual intervention for versions prior to specific one.

Key Features:
    - **Conditional Implementations**:
        Detects MindSpore's version at runtime to switch between native functions and custom equivalents.
    - **Operator Coverage**:
        [2024/09/04]
        - **view_as_complex**: Always custom due to framework limitations.
        [2024/09/02]
        - **interpolate**: mint interface post 2.3.0; ops.interpolate for earlier versions.
        [2024/07/26]
        - **conv_transpose1d**: Always custom due to framework limitations.
        - **conv_transpose2d**: Native post 2.3.0; custom for earlier versions.
        - **group_norm**: Native post 2.3.0; custom for earlier versions.
        - **multinomial**: Native post 2.4.1; custom for earlier versions.
        - **pad**: Native post 2.3.0; custom for earlier versions.
        [2025/01/14]
        - **unflatten**: Always custom due to framework limitations.
        [2025/10/22]
        - **RMSNorm**: Always custom due to framework limitations.
        [2025/11/12]
        - **scaled_dot_product_attention**: Always custom due to framework limitations.
        - **DeviceMesh**: Always custom due to framework limitations.
        [2025/11/27]
        - **cartesian_prod**: Always custom due to framework limitations.

Example:
    Import this module and use the operators as you would with native MindSpore functions, with the assurance of cross-version compatibility.

    >>> from mindone.diffusers.models.layers_compat import conv_transpose2d, interpolate
    >>> # Depending on the MindSpore version, the correct implementation will be utilized.

Todo:
    - Monitor MindSpore updates for potential native support inclusion.
    - ...
"""

import math
import numbers
from typing import Optional, Union

from packaging.version import parse

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore._c_expression.amp import AmpLevel, create_amp_strategy
from mindspore.common.api import _function_forbid_reuse
from mindspore.common.initializer import One, initializer
from mindspore.ops.function.nn_func import _interploate_ext_make_tuple, _interpolate_ext_scale_factor_convert_size

from ..utils.mindspore_utils import dtype_to_eps

__all__ = [
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "group_norm",
    "interpolate",
    "unflatten",
    "upsample_nearest3d_free_interpolate",
    "multinomial",
    "pad",
    "view_as_complex",
    "unflatten",
    "RMSNorm",
    "scaled_dot_product_attention",
    "DeviceMesh",
]

MINDSPORE_VERSION = parse(ms.__version__)


# ================================================================================
# conv_transpose1d
# ================================================================================
def _conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    # Equivalence of torch.nn.functional.conv_transpose1d
    assert output_padding == 0, "Only support output_padding == 0 so far."

    if isinstance(stride, int):
        stride = (1, stride)
    elif isinstance(stride, tuple):
        stride = (1, stride[0])

    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif isinstance(dilation, tuple):
        dilation = (dilation[0], dilation[0])

    if isinstance(padding, int):
        padding = (0, 0, padding, padding)
    elif isinstance(padding, tuple):
        padding = (0, 0, padding[0], padding[0])

    # InferShape manually
    # Format adapted from https://pytorch.org/docs/stable/generated/torch.nn.functional.conv_transpose1d.html
    input = input.unsqueeze(2)
    weight = weight.unsqueeze(2)
    batch_size, in_channels, iH, iW = input.shape
    _, out_channels_divide_groups, kH, kW = weight.shape

    out_channels = out_channels_divide_groups * groups
    outH = (iH - 1) * stride[0] - (padding[0] + padding[1]) + dilation[0] * (kH - 1) + 1
    outW = (iW - 1) * stride[1] - (padding[2] + padding[3]) + dilation[1] * (kW - 1) + 1

    # todo: unavailable mint interface
    op_conv_transpose2d = ops.Conv2DTranspose(
        out_channel=out_channels,
        kernel_size=(kH, kW),
        pad_mode="pad",
        pad=padding,
        stride=stride,
        dilation=dilation,
        group=groups,
    )
    outputs = op_conv_transpose2d(input, weight.to(input.dtype), (batch_size, out_channels, outH, outW)).squeeze(2)

    if bias is not None:
        assert isinstance(bias, ms.Tensor) and bias.ndim == 1
        bias = bias.reshape(1, -1, 1)
        outputs += bias

    return outputs


conv_transpose1d = _conv_transpose1d


# ================================================================================
# conv_transpose2d
# ================================================================================
def _conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    # Equivalence of torch.nn.functional.conv_transpose2d
    assert output_padding == 0, "Only support output_padding == 0 so far."

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (
            padding[0],
            padding[0],
            padding[1],
            padding[1],
        )

    # InferShape manually
    # Format adapted from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
    batch_size, in_channels, iH, iW = input.shape
    _, out_channels_divide_groups, kH, kW = weight.shape

    out_channels = out_channels_divide_groups * groups
    outH = (iH - 1) * stride[0] - (padding[0] + padding[1]) + dilation[0] * (kH - 1) + 1
    outW = (iW - 1) * stride[1] - (padding[2] + padding[3]) + dilation[1] * (kW - 1) + 1

    op_conv_transpose2d = ops.Conv2DTranspose(
        out_channel=out_channels,
        kernel_size=(kH, kW),
        pad_mode="pad",
        pad=padding,
        stride=stride,
        dilation=dilation,
        group=groups,
    )
    outputs = op_conv_transpose2d(input, weight.to(input.dtype), (batch_size, out_channels, outH, outW))

    if bias is not None:
        assert isinstance(bias, ms.Tensor) and bias.ndim == 1
        bias = bias.reshape(1, -1, 1, 1)
        outputs += bias

    return outputs


if MINDSPORE_VERSION >= parse("2.3.0"):
    conv_transpose2d = ms.mint.nn.functional.conv_transpose2d
else:
    conv_transpose2d = _conv_transpose2d


# ================================================================================
# conv_transpose3d
# ================================================================================
def _conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    # Equivalence of torch.nn.functional.conv_transpose3d
    # from https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html#torch.nn.ConvTranspose3d

    assert output_padding == 0, "only support output_padding == 0 so far."
    assert padding == 0, "do not support padding so far, fixme at mindspore 2.7"  # FIXME

    _, in_channels, _, _, _ = input.shape
    _, out_channels_divide_groups, kD, kH, kW = weight.shape

    assert in_channels % groups == 0, "`in_channels` should be divisible by `groups`"
    out_channels = out_channels_divide_groups * groups  # noqa F841
    in_channels_divide_groups = in_channels // groups

    if bias is not None:
        assert isinstance(bias, ms.Tensor) and bias.ndim == 1

    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    # FIXME ops.Conv3DTranspose currently supports group=1 and bias=None on ascend,
    # here we manually implement groupping and bias
    op_conv_transpose3d = ops.Conv3DTranspose(
        in_channel=in_channels_divide_groups,
        out_channel=out_channels_divide_groups,
        kernel_size=(kD, kH, kW),
        stride=stride,
        dilation=dilation,
        group=1,
    )

    input_groups = mint.chunk(input, groups, dim=1)
    weight_groups = mint.chunk(weight, groups, dim=0)
    output_groups = []
    for i in range(groups):
        # only support ms.float16 on ascend
        original_dtype = input.dtype
        output = op_conv_transpose3d(
            input_groups[i].to(ms.float16),
            weight_groups[i].to(ms.float16),
        ).to(original_dtype)

        if bias is not None:
            output += bias[i * out_channels_divide_groups : (i + 1) * out_channels_divide_groups].reshape(
                1, -1, 1, 1, 1
            )

        output_groups.append(output)

    return mint.cat(output_groups, dim=1)


conv_transpose3d = _conv_transpose3d


# ================================================================================
# group_norm
# ================================================================================
def _group_norm(x, num_groups, weight, bias, eps):
    x_shape = x.shape
    x_dtype = x.dtype

    # Calculate var&mean in float32 to avoid overflow
    x = x.reshape(x_shape[0], num_groups, -1).float()
    mean = ops.mean(x, axis=-1, keep_dims=True)
    var = ops.mean(ops.square(x - mean), axis=-1, keep_dims=True)
    x = (x - mean) / ops.sqrt(var + eps)
    x = x.reshape(x_shape).to(x_dtype)

    if weight is not None and bias is not None:
        expanded_shape = (1, -1) + (1,) * len(x_shape[2:])
        x = x * weight.reshape(expanded_shape) + bias.reshape(expanded_shape)

    return x


if MINDSPORE_VERSION >= parse("2.3.0"):
    group_norm = ms.mint.nn.functional.group_norm
else:
    group_norm = _group_norm


# ================================================================================
# nn.GELU
# ================================================================================
class _GELU(nn.Cell):
    """
    Adapted from `torch.nn.GELU`
    (https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py)
    """

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return mint.nn.functional.gelu(input, approximate=self.approximate)


if MINDSPORE_VERSION >= parse("2.6.0"):
    GELU = mint.nn.GELU
else:
    GELU = _GELU


# ================================================================================
# interpolate
# ================================================================================
if MINDSPORE_VERSION >= parse("2.3.0"):
    interpolate = mint.nn.functional.interpolate
else:
    interpolate = ops.interpolate


# ================================================================================
# upsample_nearest3d_free_interpolate
# ================================================================================
def upsample_nearest3d_free_interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    r"""
    When input is 5-dimensions tensor and mode is 'nearest', interpolate calls `aclnnUpsampleNearest3d`
    in Ascend, which is slow and doesn't support Bfloat16.

    This is an equivalent impl which doesn't use UpsampleNearest3d, it uses UpsampleNearest1d and UpsampleNearest2d
    to do the same thing.
    """
    if input.ndim != 5 or mode != "nearest":
        return interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor)

    # check for size and scale_factor
    if size is not None and scale_factor is not None:
        raise ValueError("For 'interpolate', 'size' and 'scale_factor' cannot be set simultaneously")
    if size is not None:
        size = _interploate_ext_make_tuple(input, size)
    elif scale_factor is not None:
        scale_factor = _interploate_ext_make_tuple(input, scale_factor)
        size = _interpolate_ext_scale_factor_convert_size(input, scale_factor)
        scale_factor = None
    else:
        raise ValueError("For 'interpolate', 'size' and 'scale_factor' cannot be both empty")

    B, C, T, H, W = input.shape
    # interpolate H, W
    x = interpolate(input.reshape(-1, T, H, W), size[1:])
    # interpolate T
    x = x.permute(0, 2, 3, 1).reshape(B * C, -1, T)
    x = interpolate(x, size[0])
    # reshape to (b, c, t', h', w')
    x = x.reshape(B, C, size[-2], size[-1], size[0]).permute(0, 1, 4, 2, 3)
    return x


# ================================================================================
# multinomial
# ================================================================================
@_function_forbid_reuse
def _multinomial(input, num_samples, replacement=True, **kwargs):
    assert isinstance(input, ms.Tensor) and input.ndim in (
        1,
        2,
    ), "argument input should be a MindSpore Tensor with 1 or 2 dim."
    assert (
        replacement or num_samples <= input.shape[-1]
    ), "cannot sample n_sample > prob_dist.size(-1) samples without replacement."

    input = input.float()
    input /= input.sum(-1, keepdims=True)

    if num_samples == 1 or not replacement:
        # The algorithm is from gumbel softmax.
        # s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
        # Here we can apply exp to the formula which will not affect result of
        # argmax or topk. Then we have
        # s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
        # We can also simplify the formula above by
        # s = argmax( p / q ) where q ~ Exp(1)
        # No proper Exp generator op in MindSpore,
        # so we still generate it by -log(eps)
        q = -ops.log(ops.rand_like(input))
        if num_samples == 1:
            result = (input / q).argmax(-1, keepdim=True)
        else:
            _, result = ops.topk(input / q, k=num_samples, dim=-1)
    else:
        # To generate scalar random variable X with cumulative distribution ms.mint.nn.functional(x)
        # just let X = ms.mint.nn.functional^(-1)(U) where U ~ U(0, 1)
        input = input.cumsum(-1).expand_dims(-1)
        rshape = (1, num_samples) if input.ndim == 2 else (input.shape[0], 1, num_samples)
        rand = ops.rand(*rshape, dtype=input.dtype)
        result = ops.ge(rand, input).long().sum(-2)

    return result.long()


if MINDSPORE_VERSION >= parse("2.4.1"):
    multinomial = mint.multinomial
else:
    multinomial = _multinomial


# ================================================================================
# pad
# ================================================================================
def _pad(input, pad, mode="constant", value=0):
    assert mode in ["constant", "replicate", "reflect"], "Unsupported padding mode"

    padding = [0, 0, 0, 0]
    if isinstance(pad, tuple):
        assert len(pad) <= 4, "Only support padding for the lastest 2 dimensions."
        pad = list(pad)
    padding[: len(pad)] = pad

    left, right, top, bottom = padding

    height, width = input.shape[-2:]
    other_dimensions = input.shape[:-2]
    input = input.reshape(-1, height, width)
    batch_size = input.shape[0]

    padded_height = height + top + bottom
    padded_width = width + left + right

    output = ops.full((batch_size, padded_height, padded_width), value, dtype=input.dtype)
    output[:, top : top + height, left : left + width] = input

    if mode == "replicate":
        if top > 0:
            output[:, :top, left : left + width] = input[:, 0:1, :].broadcast_to((batch_size, top, width))
        if bottom > 0:
            output[:, top + height :, left : left + width] = input[:, -1:, :].broadcast_to((batch_size, bottom, width))
        if left > 0:
            output[:, :, :left] = output[:, :, left : left + 1].broadcast_to((batch_size, padded_height, left))
        if right > 0:
            output[:, :, left + width :] = output[:, :, left + width - 1 : left + width].broadcast_to(
                (batch_size, padded_height, right)
            )
    elif mode == "reflect":
        if top > 0:
            output[:, :top, left : left + width] = (
                input[:, 1 : top + 1, :].flip(dims=[1]).broadcast_to((batch_size, top, width))
            )
        if bottom > 0:
            output[:, top + height :, left : left + width] = (
                input[:, -bottom - 1 : -1, :].flip(dims=[1]).broadcast_to((batch_size, bottom, width))
            )
        if left > 0:
            output[:, :, :left] = (
                output[:, :, left + 1 : 2 * left + 1].flip(dims=[2]).broadcast_to((batch_size, padded_height, left))
            )
        if right > 0:
            right_edge = max(0, left + width - right - 2)
            output[:, :, left + width :] = output[:, :, left + width - 2 : right_edge : -1].broadcast_to(
                (batch_size, padded_height, right)
            )

    target_shape = tuple(other_dimensions) + (padded_height, padded_width)
    output = output.reshape(*target_shape)
    return output


if MINDSPORE_VERSION >= parse("2.3.0"):
    pad = ms.mint.nn.functional.pad
else:
    pad = _pad


# ================================================================================
# view_as_complex
# ================================================================================
def _view_as_complex(input: ms.Tensor) -> ms.Tensor:
    r"""
    Equivalence of `torch.view_as_complex`.

    Args:
        input (ms.Tensor): the input tensor.

    Example:

        >>> import mindspore as ms
        >>> x = ms.ops.randn(4, 2)
        >>> x
        [[ 1.6116, -0.5772]
         [-1.4606, -0.9120]
         [ 0.0786, -1.7497]
         [-0.6561, -1.6623]]
        >>> view_as_complex(x)
        [1.6116-0.5772j   -1.4606-0.9120j   0.0786-1.7497j   -0.6561-1.6623j]
    """
    assert input.shape[-1] == 2, "Tensor must have a last dimension of size 2"
    real_part, imag_part = input.chunk(2, dim=-1)
    # todo: unavailable mint interface ops.Complex
    output = ops.Complex()(real_part, imag_part).squeeze(axis=-1)
    return output


view_as_complex = _view_as_complex


# ================================================================================
# unflatten
# ================================================================================
def _unflatten(input, dim, sizes):
    """
    Equivalence of `torch.unflatten`

    Args:
        tensor (ms.Tensor): The input tensor to unflatten.
        dim (int): The dimension to unflatten.
        sizes (tuple[int]): The target shape for the specified dimension.

    Returns:
        Tensor: A tensor with the specified dimension unflattened into the target shape.

    Raises:
        ValueError: If the specified dimension is out of range or if the product
                    of sizes does not match the size of the given dimension.
    """
    shape = input.shape

    dim = dim if dim >= 0 else dim + input.ndim

    # check validation of dim
    if dim < 0 or dim >= len(shape):
        raise ValueError(f"Invalid dimension {dim} for tensor with shape {input.shape}")

    # Calculate the product of sizes, excluding -1
    sizes_prod = 1
    num_unknown = 0
    for size in sizes:
        if size == -1:
            num_unknown += 1
        else:
            sizes_prod *= size

    # If there is one unknown size, calculate it
    if num_unknown == 1:
        sizes = tuple(size if size != -1 else shape[dim] // sizes_prod for size in sizes)

    new_shape = shape[:dim] + sizes + shape[dim + 1 :]

    return input.reshape(new_shape)


unflatten = _unflatten


# ================================================================================
# set_amp_strategy
# ================================================================================
def set_amp_strategy(net, weight_dtype=None, level=AmpLevel.AmpO3, white_list=None, black_list=None):
    """
    Apply AMP (Automatic Mixed Precision) strategy to a MindSpore network.

    Args:
        net (Cell): The neural network to configure.
        weight_dtype (ms.dtype): The target data type for weights (e.g., ms.float16).
        level (AmpLevel): The AMP level to use (e.g., AmpLevel.AmpO3).
        white_list (list): List of layer names or modules to skip casting.
        black_list (list): List of layer names or modules to explicitly cast.
    """
    if white_list is None:
        white_list = []
    if black_list is None:
        black_list = []

    net.amp_strategy = create_amp_strategy(level, weight_dtype, white_list, black_list)


# ================================================================================
# RMSNorm
# ================================================================================
class RMSNorm(nn.Cell):
    """
    Equivalence of `torch.nn.RMSNorm`
    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: Union[int, list[int], tuple[int, ...]],
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        dtype=None,
    ) -> None:
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = ms.Parameter(mint.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            self.weight.set_data(initializer(One(), self.weight.shape, self.weight.dtype))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """
        Runs the construct pass.
        """
        res = x.float() * mint.rsqrt(
            mint.pow(x.float(), 2).mean(-1, keepdim=True)
            + (self.eps if self.eps is not None else dtype_to_eps(x.dtype))
        )
        res = res.to(x.dtype) * self.weight
        return res

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


# ================================================================================
# DeviceMesh
# ================================================================================
class DeviceMesh:
    def __init__(self, device_type, mesh_shape, mesh_dim_names):
        self.device_type = device_type
        self.mesh_shape = mesh_shape
        self.mesh_dim_names = mesh_dim_names

        dim0, dim1 = mesh_shape

        self.mesh = [[r + i * dim1 for r in range(dim1)] for i in range(dim0)]
        self.groups = {}

        current_rank = mint.distributed.get_rank()
        col_groups = [[self.mesh[r][c] for r in range(dim0)] for c in range(dim1)]

        self.groups[mesh_dim_names[0]] = next(
            (mint.distributed.new_group(ranks=r) for r in col_groups if current_rank in r), None
        )
        self.groups[mesh_dim_names[1]] = next(
            (mint.distributed.new_group(ranks=r) for r in self.mesh if current_rank in r), None
        )

    def __getitem__(self, dim_name):
        return self.groups[dim_name]

    def _flatten(self):
        flat_ranks = [rank for row in self.mesh for rank in row]
        return mint.distributed.new_group(ranks=flat_ranks)


# ================================================================================
# scaled_dot_product_attention
# ================================================================================
def scaled_dot_product_attention(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: Optional[bool] = False,
    backend: Optional[str] = "flash",
):
    head_dim = query.shape[-1]

    # Note: PyTorch's SDPA and MindSpore's FA handle `attention_mask` slightly differently.
    # In PyTorch, if the mask is not boolean (e.g., float32 with 0/1 values), it is interpreted
    # as an additive bias: `attn_bias = attn_mask + attn_bias`.
    # This implicit branch may lead to issues if the pipeline mistakenly provides
    # a 0/1 float mask instead of a boolean mask.
    # While this behavior is consistent with HF Diffusers for now,
    # it may still be a potential bug source worth validating.
    if (
        (attn_mask is not None and attn_mask.dtype != ms.bool_ and 1.0 in attn_mask)
        or head_dim > 512
        or backend == "math"
        or enable_gqa
    ):
        out = math_attention_op(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    if query.dtype in (ms.float16, ms.bfloat16):
        out = flash_attention_op(query, key, value, attn_mask, keep_prob=1 - dropout_p, scale=scale)
    else:
        out = flash_attention_op(
            query.to(ms.float16),
            key.to(ms.float16),
            value.to(ms.float16),
            attn_mask,
            keep_prob=1 - dropout_p,
            scale=scale,
        ).to(query.dtype)
    return out


def math_attention_op(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = mint.zeros((L, S), dtype=query.dtype)
    if is_causal:
        if attn_mask is not None:
            if attn_mask.dtype == ms.bool_:
                attn_mask = mint.logical_and(attn_mask, mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0))
            else:
                attn_mask = attn_mask + mint.triu(mint.full((L, S), float("-inf"), dtype=attn_mask.dtype), diagonal=1)
        else:
            temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.shape[-3] // key.shape[-3], -3)
        value = value.repeat_interleave(query.shape[-3] // value.shape[-3], -3)

    attn_weight = mint.matmul(query, key.swapaxes(-2, -1)) * scale_factor
    attn_weight += attn_bias
    attn_weight = mint.softmax(attn_weight, dim=-1)
    attn_weight = ops.dropout(attn_weight, dropout_p, training=True)
    return mint.matmul(attn_weight, value)


def flash_attention_op(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    keep_prob: float = 1.0,
    scale: Optional[float] = None,
):
    # For most scenarios, qkv has been processed into a BNSD layout before sdp
    input_layout = "BNSD"
    head_num = query.shape[1]
    if scale is None:
        scale = query.shape[-1] ** (-0.5)

    # In case qkv is 3-dim after `head_to_batch_dim`
    if query.ndim == 3:
        input_layout = "BSH"
        head_num = 1

    # process `attn_mask` as logic is different between PyTorch and Mindspore
    # In MindSpore, False indicates retention and True indicates discard, in PyTorch it is the opposite
    if attn_mask is not None:
        attn_mask = mint.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
        attn_mask = mint.broadcast_to(
            attn_mask, (attn_mask.shape[0], attn_mask.shape[1], query.shape[-2], key.shape[-2])
        )[:, :1, :, :]

    return ops.operations.nn_ops.FlashAttentionScore(
        head_num=head_num, keep_prob=keep_prob, scale_value=scale, input_layout=input_layout
    )(query, key, value, None, None, None, attn_mask)[3]


def cartesian_prod(*tensors):
    """
    Equivalence of `torch.cartesian_prod`
    """
    grids = mint.meshgrid(*tensors, indexing="ij")
    return mint.stack(tuple(grid.reshape(-1) for grid in grids), dim=1)
