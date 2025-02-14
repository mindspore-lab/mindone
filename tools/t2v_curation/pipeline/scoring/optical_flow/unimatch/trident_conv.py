import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeUniform, Zero

class MultiScaleTridentConv(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        strides=1,
        paddings=0,
        dilations=1,
        dilation=1,
        group=1,
        num_branch=1,
        test_branch_idx=-1,
        has_bias=False,
        norm=None,
        activation=None,
    ):
        super(MultiScaleTridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.num_branch = num_branch
        self.group = group
        self.has_bias = has_bias
        self.dilation = dilation

        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        if isinstance(strides, int):
            strides = [strides] * self.num_branch
        self.paddings = [padding if isinstance(padding, tuple) else (padding, padding) for padding in paddings]
        self.dilations = [dilation if isinstance(dilation, tuple) else (dilation, dilation) for dilation in dilations]
        self.strides = [stride if isinstance(stride, tuple) else (stride, stride) for stride in strides]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation

        assert len({self.num_branch, len(self.paddings), len(self.strides)}) == 1

        weight_shape = [out_channels, in_channels // group, *self.kernel_size]
        self.weight = ms.Parameter(
            ms.common.initializer.initializer(HeUniform(), weight_shape, ms.float32), name='weight'
        )
        if has_bias:
            self.bias = ms.Parameter(
                ms.common.initializer.initializer(Zero(), [out_channels], ms.float32), name='bias'
            )
        else:
            self.bias = None

    def construct(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        outputs = []
        if self.training or self.test_branch_idx == -1:
            for idx, (input_tensor, padding, stride, dilation) in enumerate(
                    zip(inputs, self.paddings, self.strides, self.dilations)
            ):
                padded_input = ops.Pad(((0, 0), (0, 0), padding, padding))(input_tensor)
                output = ops.conv2d(
                    padded_input,
                    self.weight,
                    self.bias,
                    stride=stride,
                    pad_mode='pad',
                    padding=0,  # pad above, no padding here
                    dilation=dilation,
                    groups=self.group,
                )
                outputs.append(output)
        else:
            idx = self.test_branch_idx if self.test_branch_idx != -1 else -1
            input_tensor = inputs[0]
            padding = self.paddings[idx]
            stride = self.strides[idx]
            dilation = self.dilations[idx]
            padded_input = ops.Pad(((0, 0), (0, 0), padding, padding))(input_tensor)
            output = ops.conv2d(
                padded_input,
                self.weight,
                self.bias,
                stride=stride,
                pad_mode='pad',
                padding=0,
                dilation=dilation,
                groups=self.group,
            )
            outputs.append(output)

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]

        return outputs