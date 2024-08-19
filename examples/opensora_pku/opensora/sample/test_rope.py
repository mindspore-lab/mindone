import torch

torch_npu = None
npu_config = None
get_sequence_parallel_state = False
import numpy as np
from opensora.models.diffusion.opensora.modules import PositionGetter3D as PositionGetter3D_MS
from opensora.models.diffusion.opensora.modules import RoPE3D as RoPE3D_MS

import mindspore as ms


class PositionGetter3D(object):
    """return positions of patches"""

    def __init__(
        self,
    ):
        self.cache_positions = {}

    def __call__(self, b, t, h, w, device):
        if not (b, t, h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            z = torch.arange(t, device=device)
            pos = torch.cartesian_prod(z, y, x)
            if get_sequence_parallel_state:
                # print('PositionGetter3D', PositionGetter3D)
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, -1, 1).contiguous().expand(3, -1, b).clone()
            else:
                pos = pos.reshape(t * h * w, 3).transpose(0, 1).reshape(3, 1, -1).contiguous().expand(3, b, -1).clone()
            poses = (pos[0].contiguous(), pos[1].contiguous(), pos[2].contiguous())
            max_poses = (int(poses[0].max()), int(poses[1].max()), int(poses[2].max()))

            self.cache_positions[b, t, h, w] = (poses, max_poses)
        pos = self.cache_positions[b, t, h, w]

        return pos


class RoPE3D(torch.nn.Module):
    def __init__(self, freq=10000.0, F0=1.0, interpolation_scale_thw=(1, 1, 1)):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.interpolation_scale_t = interpolation_scale_thw[0]
        self.interpolation_scale_h = interpolation_scale_thw[1]
        self.interpolation_scale_w = interpolation_scale_thw[2]
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype, interpolation_scale=1):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype) / interpolation_scale
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        if npu_config is None and not get_sequence_parallel_state:
            # for (batch_size x nheads x ntokens x dim)
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        else:
            # for (batch_size x ntokens x nheads x dim)
            cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]

        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (t, y and x position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x x dim)
        """
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        poses, max_poses = positions
        assert len(poses) == 3 and poses[0].ndim == 2  # Batch, Seq, 3
        cos_t, sin_t = self.get_cos_sin(D, max_poses[0] + 1, tokens.device, tokens.dtype, self.interpolation_scale_t)
        cos_y, sin_y = self.get_cos_sin(D, max_poses[1] + 1, tokens.device, tokens.dtype, self.interpolation_scale_h)
        cos_x, sin_x = self.get_cos_sin(D, max_poses[2] + 1, tokens.device, tokens.dtype, self.interpolation_scale_w)
        # split features into three along the feature dimension, and apply rope1d on each half
        t, y, x = tokens.chunk(3, dim=-1)
        t = self.apply_rope1d(t, poses[0], cos_t, sin_t)
        y = self.apply_rope1d(y, poses[1], cos_y, sin_y)
        x = self.apply_rope1d(x, poses[2], cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1)
        return tokens


if __name__ == "__main__":
    ms.set_context(mode=1)

    query = np.load("ms-res-0815/attn1_query.npy")
    key = np.load("ms-res-0815/attn1_key.npy")
    value = np.load("ms-res-0815/attn1_value.npy")

    batch_size = 1
    frame = 8
    height, width = 45, 80
    interpolation_scale_thw = (1.0, 1.5, 2.0)
    # torch
    position_getter = PositionGetter3D()
    rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw)
    query_torch = torch.tensor(query, dtype=torch.bfloat16).view(batch_size, -1, 24, 96).transpose(1, 2)
    key_torch = torch.tensor(key, dtype=torch.bfloat16).view(batch_size, -1, 24, 96).transpose(1, 2)

    pos_thw_torch = position_getter(batch_size, t=frame, h=height, w=width, device=torch.device("cpu"))
    query_torch = rope(query_torch, pos_thw_torch).float().numpy()
    key_torch = rope(key_torch, pos_thw_torch).float().numpy()

    # ms
    position_getter = PositionGetter3D_MS()
    rope = RoPE3D_MS(interpolation_scale_thw=interpolation_scale_thw)
    query_ms = ms.Tensor(query, dtype=ms.bfloat16).view(batch_size, -1, 24, 96)
    key_ms = ms.Tensor(key, dtype=ms.bfloat16).view(batch_size, -1, 24, 96)

    pos_thw_ms = position_getter(batch_size, t=frame, h=height, w=width)
    query_ms = rope(query_ms, pos_thw_ms)
    key_ms = rope(key_ms, pos_thw_ms)

    query_ms = query_ms.swapaxes(1, 2).float().asnumpy()
    key_ms = key_ms.swapaxes(1, 2).float().asnumpy()
