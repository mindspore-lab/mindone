from opensora.acceleration.parallel_states import get_sequence_parallel_state

import mindspore as ms
from mindspore import mint, nn, ops


def ms_cartesian_prod(*tensors):
    for tensor in tensors:
        assert len(tensor.shape) == 1, f"Accept 1-dim tensor as input! But got {len(tensor.shape)}-dim input"
    pools = [tuple(pool) for pool in tensors]
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    return ms.Tensor(result)


class PositionGetter3D(object):
    """return positions of patches"""

    def __init__(
        self,
    ):
        self.cache_positions = {}

    def __call__(self, b, t, h, w):
        if not (b, t, h, w) in self.cache_positions:
            x = ops.arange(w)
            y = ops.arange(h)
            z = ops.arange(t)
            pos = ms_cartesian_prod(z, y, x)
            if get_sequence_parallel_state():
                # print('PositionGetter3D', PositionGetter3D)
                pos = pos.reshape(t * h * w, 3).swapaxes(0, 1).reshape(3, -1, 1).broadcast_to(3, -1, b)
            else:
                pos = pos.reshape(t * h * w, 3).swapaxes(0, 1).reshape(3, 1, -1).broadcast_to(3, b, -1)
            poses = (pos[0], pos[1], pos[2])
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]

        return pos


class RoPE3D(nn.Cell):
    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, D, seq_len, dtype, interpolation_scale=1):
        if (D, seq_len, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (ops.arange(0, D, 2).float() / D))
            t = ops.arange(seq_len, dtype=inv_freq.dtype) / interpolation_scale
            freqs = ms.numpy.outer(t, inv_freq).to(dtype)
            freqs = ops.cat((freqs, freqs), axis=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, dtype] = (cos, sin)
        return self.cache[D, seq_len, dtype]

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
        cos = mint.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = mint.nn.functional.embedding(pos1d, sin)[:, :, None, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def construct(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x x dim)
        """
        assert tokens.shape[3] % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.shape[3] // 3
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2  # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D, max_poses[0] + 1, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, max_poses[1] + 1, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, max_poses[2] + 1, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        # t, y, x = tokens.chunk(3, dim=-1)
        t, y, x = mint.chunk(tokens, axis=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = ops.cat((t, y, x), axis=-1)
        return tokens
