import mindspore as ms
from mindspore import nn


class GroupNormExtend(nn.GroupNorm):
    # GroupNorm supporting tensors with more than 4 dim
    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 5:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x)
        return y.view(x_shape)


def Normalize(in_channels, num_groups=32, extend=False):
    if extend:
        return GroupNormExtend(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True).to_float(
            ms.float32
        )
    else:
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True).to_float(ms.float32)
