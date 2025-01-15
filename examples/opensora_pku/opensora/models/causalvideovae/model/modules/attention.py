import logging

import mindspore as ms
from mindspore import mint, nn

from .conv import CausalConv3d
from .normalize import Normalize

try:
    from opensora.npu_config import npu_config
except ImportError:
    npu_config = None

_logger = logging.getLogger(__name__)


class AttnBlock3D(nn.Cell):
    """Compatible with old versions, there are issues, use with caution."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.bmm = mint.bmm
        self.softmax = mint.nn.Softmax(dim=2)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = q.reshape(b * t, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b * t, c, h * w)  # b,c,hw
        w_ = self.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = self.softmax(w_)

        # attend to values
        v = v.reshape(b * t, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = self.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, t, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3DFix(nn.Cell):
    """
    Thanks to https://github.com/PKU-YuanGroup/Open-Sora-Plan/pull/172.
    """

    def __init__(self, in_channels, norm_type="groupnorm", dtype=ms.float32):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype

        self.norm = Normalize(in_channels, norm_type=norm_type)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def construct(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)

        if npu_config.enable_FA and q.dtype == ms.float32:
            dtype = ms.bfloat16
        else:
            dtype = None
        npu_config.current_run_dtype = dtype
        npu_config.original_run_dtype = q.dtype
        # with set_run_dtype(q, dtype): # graph mode does not support it
        query, key, value = npu_config.set_current_run_dtype([q, k, v])
        hidden_states = npu_config.run_attention(
            query,
            key,
            value,
            attention_mask=None,
            input_layout="BNSD",
            head_dim=c // 2,
            head_num=2,  # FIXME: different from torch. To make head_dim 256 instead of 512
        )
        npu_config.current_run_dtype = None
        npu_config.original_run_dtype = None
        attn_output = npu_config.restore_dtype(hidden_states)

        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        h_ = self.proj_out(attn_output)

        return x + h_
