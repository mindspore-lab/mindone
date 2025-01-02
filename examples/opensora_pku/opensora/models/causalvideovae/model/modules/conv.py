import math
from collections import deque
from typing import Tuple, Union

import mindspore as ms
from mindspore.common.initializer import HeUniform, Uniform

try:
    from opensora.npu_config import npu_config
except ImportError:
    npu_config = None

from mindspore import mint, nn, ops


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class Conv2d(nn.Conv2d):
    """
    Conv2d for video input (B C T H W)
    """

    def rearrange_in(self, x):
        # b c f h w -> b f c h w
        B, C, F, H, W = x.shape
        x = mint.permute(x, (0, 2, 1, 3, 4))
        # -> (b*f c h w)
        x = ops.reshape(x, (-1, C, H, W))

        return x

    def rearrange_out(self, x, F):
        BF, D, H_, W_ = x.shape
        # (b*f D h w) -> (b f D h w)
        x = ops.reshape(x, (BF // F, F, D, H_, W_))
        # -> (b D f h w)
        x = mint.permute(x, (0, 2, 1, 3, 4))

        return x

    def construct(self, x):
        # import pdb; pdb.set_trace()
        # x: (b c f h w)
        F = x.shape[-3]
        x = self.rearrange_in(x)

        x = super().construct(x)

        x = self.rearrange_out(x, F)

        return x


class CausalConv3d(nn.Cell):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=False,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.stride = cast_tuple(self.stride, 3)
        conv_dtype = npu_config.conv_dtype if npu_config is not None else ms.bfloat16
        if self.padding == 0:
            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                self.kernel_size,
                stride=self.stride,
                pad_mode="valid",
                has_bias=bias,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(
                    scale=1 / math.sqrt(chan_in * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
                ),
                dtype=conv_dtype,
                **kwargs,
            )
        else:
            self.padding = list(cast_tuple(self.padding, 6))
            self.padding[0] = 0
            self.padding[1] = 0

            self.conv = nn.Conv3d(
                chan_in,
                chan_out,
                self.kernel_size,
                stride=self.stride,
                padding=tuple(self.padding),
                pad_mode="pad",
                has_bias=bias,
                weight_init=HeUniform(negative_slope=math.sqrt(5)),
                bias_init=Uniform(
                    scale=1 / math.sqrt(chan_in * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
                ),
                dtype=conv_dtype,
                **kwargs,
            )
        self.enable_cached = enable_cached
        self.is_first_chunk = True

        self.causal_cached = deque()
        self.cache_offset = 0

    def construct(self, x):
        x_dtype = x.dtype
        # x: (bs, Cin, T, H, W )
        # first_frame_pad = ops.repeat_interleave(first_frame, (self.time_kernel_size - 1), axis=2)
        if self.time_kernel_size - 1 > 0:
            if self.is_first_chunk:
                first_frame = x[:, :, :1, :, :]
                first_frame_pad = mint.cat([first_frame] * (self.time_kernel_size - 1), dim=2)
                # first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))
            else:
                first_frame_pad = self.causal_cached.popleft()

            x = mint.cat((first_frame_pad, x), dim=2)

        if self.enable_cached and self.time_kernel_size != 1:
            if (self.time_kernel_size - 1) // self.stride[0] != 0:
                if self.cache_offset == 0:
                    causal_cached = x[:, :, -(self.time_kernel_size - 1) // self.stride[0] :]
                else:
                    causal_cached = x[:, :, : -self.cache_offset][
                        :, :, -(self.time_kernel_size - 1) // self.stride[0] :
                    ]
            else:
                causal_cached = x[:, :, 0:0, :, :]
            self.causal_cached.append(causal_cached.copy())
        elif self.enable_cached:
            self.causal_cached.append(x[:, :, 0:0, :, :].copy())

        if npu_config is not None and npu_config.on_npu:
            return npu_config.run_conv3d(self.conv, x, x_dtype)
        else:
            x = self.conv(x)
            return x
