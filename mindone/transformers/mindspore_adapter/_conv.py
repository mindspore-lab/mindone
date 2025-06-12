import math

import mindspore.mint as mint
import mindspore.mint.nn.functional as F
from mindspore import Parameter, Tensor
from mindspore.common.initializer import HeUniform, initializer
from mindspore.ops.function.nn_func import conv_transpose2d, pad_ext


class Conv1d(mint.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype=None,
    ):
        assert isinstance(kernel_size, int)
        assert isinstance(stride, int)
        assert isinstance(dilation, int)
        kernel_size = (1, kernel_size)
        stride = (1, stride)
        dilation = (1, dilation)

        if isinstance(padding, int):
            padding = (0, padding)
        else:
            assert isinstance(padding, str)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            dtype=dtype,
        )
        weight_init = HeUniform(math.sqrt(5))
        shape = (*self.weight.shape[:2], self.weight.shape[-1])
        self.weight = Parameter(initializer(weight_init, shape, dtype=self.weight.dtype), name="weight")

    def construct(self, x: Tensor):
        x = mint.unsqueeze(x, dim=-2)
        weight = mint.unsqueeze(self.weight, dim=-2)
        if self.padding_mode != "zeros":
            x = self.conv2d(
                pad_ext(x, self._reversed_padding, mode=self.padding_mode),
                weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        else:
            x = self.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x = mint.squeeze(x, dim=-2)
        return x


class ConvTranspose1d(mint.nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        dtype=None,
    ):
        assert isinstance(kernel_size, int)
        assert isinstance(stride, int)
        assert isinstance(dilation, int)
        kernel_size = (1, kernel_size)
        stride = (1, stride)
        dilation = (1, dilation)

        assert isinstance(padding, int)
        assert isinstance(output_padding, int)
        padding = (0, padding)
        output_padding = (0, output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            dtype=dtype,
        )
        weight_init = HeUniform(math.sqrt(5))
        shape = (*self.weight.shape[:2], self.weight.shape[-1])
        self.weight = Parameter(initializer(weight_init, shape, dtype=self.weight.dtype), name="weight")

    def construct(self, x: Tensor):
        x = mint.unsqueeze(x, dim=-2)
        weight = mint.unsqueeze(self.weight, dim=-2)
        num_spatial_dims = 2
        output_padding = self._output_padding(
            x,
            None,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )

        x = conv_transpose2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        x = mint.squeeze(x, dim=-2)
        return x


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert isinstance(stride, int)
    assert isinstance(dilation, int)

    stride = (1, stride)
    dilation = (1, dilation)
    if isinstance(padding, int):
        padding = (0, padding)
    else:
        assert isinstance(padding, str)

    input = mint.unsqueeze(input, dim=-2)
    weight = mint.unsqueeze(weight, dim=-2)
    output = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    output = mint.squeeze(output, dim=-2)
    return output


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    assert isinstance(stride, int)
    assert isinstance(dilation, int)
    assert isinstance(padding, int)
    assert isinstance(output_padding, int)
    stride = (1, stride)
    dilation = (1, dilation)
    padding = (0, padding)
    output_padding = (0, output_padding)

    input = mint.unsqueeze(input, dim=-2)
    weight = mint.unsqueeze(weight, dim=-2)
    output = F.conv_transpose2d(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    output = mint.squeeze(output, dim=-2)
    return output
