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
        [2024/07/26]
        - **conv_transpose1d**: Always custom due to framework limitations.
        - **conv_transpose2d**: Native post 2.3.0; custom for earlier versions.
        - **group_norm**: Native post 2.3.0; custom for earlier versions.
        - **multinomial**: Native post 2.3.0; custom for earlier versions.
        - **pad**: Native post 2.3.0; custom for earlier versions.

Example:
    Import this module and use the operators as you would with native MindSpore functions, with the assurance of cross-version compatibility.

    >>> from mindone.diffusers.models.layers_compat import conv_transpose2d, interpolate
    >>> # Depending on the MindSpore version, the correct implementation will be utilized.

Todo:
    - Monitor MindSpore updates for potential native support inclusion.
    - ...
"""

from packaging.version import parse

import mindspore as ms
from mindspore import ops
from mindspore.common.api import _function_forbid_reuse

__all__ = [
    "conv_transpose1d",
    "conv_transpose2d",
    "group_norm",
    "multinomial",
    "pad",
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
# group_norm
# ================================================================================
def _group_norm(x, num_groups, weight, bias, eps):
    x_shape = x.shape
    x = x.reshape(x_shape[0], num_groups, -1)
    var, mean = ops.var_mean(x, axis=-1, keepdims=True)
    x = (x - mean) / ops.sqrt(var + eps)
    x = x.reshape(x_shape)

    if weight is not None and bias is not None:
        expanded_shape = (1, -1) + (1,) * len(x_shape[2:])
        x = x * weight.reshape(expanded_shape) + bias.reshape(expanded_shape)

    return x


if MINDSPORE_VERSION >= parse("2.3.0"):
    group_norm = ms.mint.nn.functional.group_norm
else:
    group_norm = _group_norm


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


if MINDSPORE_VERSION >= parse("2.3.0"):
    multinomial = ops.multinomial
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
