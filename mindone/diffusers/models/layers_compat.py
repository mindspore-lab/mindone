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
        - **fp32_interpolate**: Always custom (upcast to fp32 during computation and cast back after)
                                due to origin interface doesn't supported bfloat16 data type.
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
from mindspore.ops.function.nn_func import _interploate_ext_make_tuple, _interpolate_ext_scale_factor_convert_size

__all__ = [
    "conv_transpose1d",
    "conv_transpose2d",
    "group_norm",
    "interpolate",
    "fp32_interpolate",
    "upsample_nearest3d_free_interpolate",
    "multinomial",
    "pad",
    "view_as_complex",
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
# interpolate
# ================================================================================
if MINDSPORE_VERSION >= parse("2.3.0"):
    interpolate = ms.mint.nn.functional.interpolate
else:
    interpolate = ops.interpolate


# ================================================================================
# FP32 interpolate
# ================================================================================
def _fp32_interpolate(input, **kwargs):
    r"""
    Samples the input Tensor to the given size or scale_factor by using one of the interpolate algorithms.
    Upcast to float32 for computation and re-cast back to original data type as BFloat16 is not supported
    for interpolate algorithms.

    .. note::
        - In 'linear' mode, backpropagation does not support scenarios where `scale_factor` is not None
          and `align_corners` is False.

    Args:
        input (Tensor): Tensor to be resized.
            Input tensor must be a 3-D, 4-D, or 5-D tensor with shape
            :math:`(N, C, [optional D], [optional H], W)` , with data type of float.
        size (Union[int, tuple[int], list[int]], optional): The target size.
            If size is a tuple or list, its length should be the same as the number of dimensions in input
            after removing the first two dimensions N, C.
            One and only one of size and scale_factor can be set to None. Default: ``None`` .
        scale_factor (Union[float, tuple[float], list[float]], optional): The scale factor of new size of the tensor.
            If scale_factor is a tuple or list, its length should be the same as the number of dimensions in input
            after removing the first two dimensions N, C.
            One and only one of size and scale_factor can be set to None. Default: ``None`` .
        mode (str): The sampling algorithm.
            One of 'nearest', 'linear' (3D only), 'bilinear' (4D only), 'trilinear' (5D only), 'bicubic' (4D only),
            'area', 'nearest-exact'(matches Scikit-Image and PIL nearest neighbours interpolation algorithms and fixes
            knows issues with `nearest`, 3D and 4D). Default: ``"nearest"`` .

        align_corners (bool): Whether to use corner alignment for coordinate mapping. Assuming a transformation is
            applied to the input Tensor along the x-axis, the specific calculation formula is as follows:

            .. code-block::

                ori_i = new_length != 1 ? new_i * (ori_length - 1) / (new_length - 1) : 0   # 'align_corners' = True

                ori_i = new_length > 1 ? (new_i + 0.5) * ori_length / new_length - 0.5 : 0  # 'align_corners' = False

            Among them, :math:`ori\_length` and :math:`new\_length` represent the length of the Tensor before and after
            transformation along the x-axis respectively; :math:`new\_i` represents the coordinate of the i-th element
            along the x-axis after transformation; :math:`ori\_i` represents
            the corresponding coordinate of the original
            data along the x-axis.

            This is only valid for ``'linear'``, ``'bilinear'``, or ``'bicubic'`` modes. Default: ``False`` .
        recompute_scale_factor (bool, optional): Recalculate `scale_factor`.
            If True, the parameter `size` will be calculated using the value of the `scale_factor`,
            and finally scaled using the value of `size`.
            If False, the value of `size` or `scale_factor` will be used for direct interpolation. Default: ``None`` .

    .. note::
        The 'nearest-exact' mode is the same as the nearest-neighbor interpolation algorithm used in
        scikit-image and PIL. The 'nearest' mode produces the same results as the INTER_NEAREST interpolation
        algorithm used in OpenCV.

    Args Support List and Supported Platforms:

    +---------------+-----------+---------------+--------------+----------------+
    | mode          | input.dim | align_corners | scale_factor | device         |
    +===============+===========+===============+==============+================+
    | nearest       | 3         | \-            | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    |               | 4         | \-            | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    |               | 5         | \-            | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    | linear        | 3         | √             | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    | bilinear      | 4         | √             | ×            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    | bicubic       | 4         | √             | ×            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    | area          | 3         | \-            | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    |               | 4         | \-            | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    |               | 5         | \-            | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+
    | nearest-exact | 3         | \-            | ×            | Ascend,CPU     |
    +---------------+-----------+---------------+--------------+----------------+
    |               | 4         | \-            | ×            | Ascend,CPU     |
    +---------------+-----------+---------------+--------------+----------------+
    | trilinear     | 5         | √             | √            | Ascend,GPU,CPU |
    +---------------+-----------+---------------+--------------+----------------+

    - `-` indicates that there is no such parameter.
    - `×` indicates that this parameter is not currently supported.
    - `√` indicates that this parameter is supported.

    Returns:
        Tensor, resized, whose dimensions and dtype are the same as `input`.
    """
    input_dtype = input.dtype
    output = interpolate(input.float(), **kwargs).to(input_dtype)
    return output


fp32_interpolate = _fp32_interpolate


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


# ================================================================================
# view_as_complex
# ================================================================================
def _view_as_complex(input: ms.Tensor) -> ms.Tensor:
    r"""
    view_as_complex(input) -> Tensor

    Equivalence of `torch.view_as_complex`. Returns a view of :attr:`input` as a
    complex tensor. For an input complex tensor of :attr:`size` :math:`m1, m2, \dots,
    mi, 2`, this function returns a new complex tensor of :attr:`size` :math:`m1, m2,
    \dots, mi` where the last dimension of the input tensor is expected to represent
    the real and imaginary components of complex numbers.

    .. warning::
        :func:`view_as_complex` is only supported for tensors with
        :class:`ms.dtype` ``ms.float64`` and ``ms.float32``.  The input is
        expected to have the last dimension of :attr:`size` 2. In addition, the
        tensor must have a `stride` of 1 for its last dimension. The strides of all
        other dimensions must be even numbers.

    Args:
        input (ms.Tensor): the input tensor.

    Example::

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
    real_part, imag_part = input.chunk(2, axis=-1)
    output = ops.Complex()(real_part, imag_part).squeeze(axis=-1)
    return output


view_as_complex = _view_as_complex
