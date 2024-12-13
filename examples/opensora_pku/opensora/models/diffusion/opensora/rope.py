import itertools

import numpy as np
from opensora.acceleration.parallel_states import get_sequence_parallel_state

import mindspore as ms
from mindspore import mint, nn, ops


class PositionGetter3D(object):
    """return positions of patches"""

    def __init__(
        self,
    ):
        pass

    def __call__(self, b, t, h, w):
        x = list(range(w))
        y = list(range(h))
        z = list(range(t))
        pos = list(itertools.product(z, y, x))
        pos = ms.Tensor(pos)
        if get_sequence_parallel_state():
            # print('PositionGetter3D', PositionGetter3D)
            pos = pos.reshape(t * h * w, 3).swapaxes(0, 1).reshape(3, -1, 1).broadcast_to((3, -1, b))
        else:
            pos = pos.reshape(t * h * w, 3).swapaxes(0, 1).reshape(3, 1, -1).broadcast_to((3, b, -1))
        poses = (pos[0], pos[1], pos[2])
        max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

        pos = (poses, max_poses)

        return pos


class RoPE3D(nn.Cell):
    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1), dim_head=64):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.dim_head = dim_head
        assert (
            self.dim_head % 3 == 0
        ), f"number of head dimensions should be a multiple of three, but got {self.dim_head}"
        D = self.dim_head // 3
        self.inv_freq = ms.Tensor(1.0 / (self.base ** (np.arange(0, D, 2, dtype=np.float64) / D)), dtype=ms.float32)
        # self.cache = {}

    def get_cos_sin(self, seq_len, interpolation_scale=1):
        t = ops.arange(seq_len, dtype=self.inv_freq.dtype) / interpolation_scale
        freqs = ops.outer(t, self.inv_freq).to(self.inv_freq.dtype)
        freqs = ops.cat((freqs, freqs), axis=-1)
        cos = freqs.cos()  # (Seq, Dim)
        sin = freqs.sin()
        return cos, sin

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return ops.cat((-x2, x1), axis=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        # if npu_config is None and not get_sequence_parallel_state():
        #     # for (batch_size x nheads x ntokens x dim)
        #     cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        #     sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        # else:
        # for (batch_size x ntokens x nheads x dim)
        cos = cos[pos1d.to(ms.int32)][:, :, None, :]
        sin = sin[pos1d.to(ms.int32)][:, :, None, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def construct(self, tokens, positions):
        """
        input:
            * tokens: batch_size x ntokens x nheads x  dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x ntokens x nheads x dim)
        """
        assert tokens.shape[3] % 3 == 0, "number of dimensions should be a multiple of three"

        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2  # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(max_poses[0] + 1, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(max_poses[1] + 1, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(max_poses[2] + 1, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        # t, y, x = tokens.chunk(3, dim=-1)
        t, y, x = mint.chunk(tokens, 3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t.to(tokens.dtype), sin_t.to(tokens.dtype))
        y = self.apply_rope1d(y, poses[1], cos_y.to(tokens.dtype), sin_y.to(tokens.dtype))
        x = self.apply_rope1d(x, poses[2], cos_x.to(tokens.dtype), sin_x.to(tokens.dtype))
        tokens = ops.cat((t, y, x), axis=-1)
        return tokens
