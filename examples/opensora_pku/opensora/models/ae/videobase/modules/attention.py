import logging

import mindspore as ms
from mindspore import nn, ops

from .conv import CausalConv3d

_logger = logging.getLogger(__name__)


class AttnBlock(nn.Cell):
    def __init__(self, in_channels, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype
        self.bmm = ops.BatchMatMul()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True).to_float(
            dtype
        )
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True).to_float(
            dtype
        )
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True).to_float(
            dtype
        )
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, pad_mode="valid", has_bias=True
        ).to_float(dtype)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = ops.reshape(q, (b, c, h * w))
        q = ops.transpose(q, (0, 2, 1))  # b,hw,c
        k = ops.reshape(k, (b, c, h * w))  # b,c,hw
        w_ = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        w_ = w_ * (int(c) ** (-0.5))
        # FIXME: cast w_ to FP32 in amp
        w_ = ops.Softmax(axis=2)(w_)

        # attend to values
        v = ops.reshape(v, (b, c, h * w))
        w_ = ops.transpose(w_, (0, 2, 1))  # b,hw,hw (first hw of k, second of q)
        h_ = self.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = ops.reshape(h_, (b, c, h, w))

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3DFix(nn.Cell):
    def __init__(self, in_channels, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype

        self.bmm = ops.BatchMatMul()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # TODO: 1x1 conv3d can be replaced with flatten and Linear
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def construct(self, x):
        # q shape: (b c t h w)
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        # q: (b c t h w) -> (b t c h w) -> (b*t c h*w) -> (b*t h*w c)
        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 1, 3, 4)
        q = ops.reshape(q, (b * t, c, h * w))
        q = q.permute(0, 2, 1)  # b,hw,c

        # k: (b c t h w) -> (b t c h w) -> (b*t c h*w)
        k = k.permute(0, 2, 1, 3, 4)
        k = ops.reshape(k, (b * t, c, h * w))

        # w: (b*t hw hw)
        # TODO: support Flash Attention
        w_ = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        # FIXME: cast w_ to FP32 in amp
        w_ = ops.Softmax(axis=2)(w_)

        # attend to values
        # v: (b c t h w) -> (b t c h w) -> (bt c hw)
        # w_: (bt hw hw) -> (bt hw hw)
        v = v.permute(0, 2, 1, 3, 4)
        v = ops.reshape(v, (b * t, c, h * w))
        w_ = ops.transpose(w_, (0, 2, 1))  # b,hw,hw (first hw of k, second of q)
        h_ = self.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        # h_: (b*t c hw) -> (b t c h w) -> (b c t h w)
        h_ = ops.reshape(h_, (b, t, c, h, w))
        h_ = h_.permute(0, 2, 1, 3, 4)

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3D(nn.Cell):
    def __init__(self, in_channels, dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype

        self.bmm = ops.BatchMatMul()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # TODO: 1x1 conv3d can be replaced with flatten and Linear
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def construct(self, x):
        # q shape: (b c t h w)
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        # q: (b c t h w) -> (b t c h w) -> (b*t c h*w) -> (b*t h*w c)
        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 1, 3, 4)
        q = ops.reshape(q, (b * t, c, h * w))
        q = q.permute(0, 2, 1)  # b,hw,c

        # k: (b c t h w) -> (b t c h w) -> (b*t c h*w)
        k = k.permute(0, 2, 1, 3, 4)
        k = ops.reshape(k, (b * t, c, h * w))

        # w: (b*t hw hw)
        # TODO: support Flash Attention
        w_ = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        # FIXME: cast w_ to FP32 in amp
        w_ = ops.Softmax(axis=2)(w_)

        # attend to values
        # v: (b c t h w) -> (b t c h w) -> (bt c hw)
        # w_: (bt hw hw) -> (bt hw hw)
        v = v.permute(0, 2, 1, 3, 4)
        v = ops.reshape(v, (b * t, c, h * w))
        w_ = ops.transpose(w_, (0, 2, 1))  # b,hw,hw (first hw of k, second of q)
        h_ = self.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        # h_: (b*t c hw) -> (b t c h w) -> (b c t h w)
        h_ = ops.reshape(h_, (b, t, c, h, w))
        h_ = h_.permute(0, 2, 1, 3, 4)

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(in_channels, attn_type="vanilla", dtype=ms.float32):
    assert attn_type in ["vanilla", "vanilla3D"], f"attn_type {attn_type} not supported"
    _logger.debug(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels, dtype=dtype)
    elif attn_type == "vanilla3D":
        return AttnBlock3D(in_channels, dtype=dtype)
