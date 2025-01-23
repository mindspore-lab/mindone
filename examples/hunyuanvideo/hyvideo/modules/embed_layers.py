import math
import numpy as np
import mindspore as ms
from mindspore import nn, ops, mint
from mindspore.common.initializer import Normal, XavierUniform, initializer
from ..utils.helpers import to_2tuple

class PatchEmbed(nn.Cell):
    def __init__(
        self,
        patch_size=2,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        use_conv2d=False,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        # print('D--: patch_size, ', patch_size)

        self.use_conv2d = use_conv2d
        if self.use_conv2d:
            print('PatchEmbed with conv2d equivalence')
        if use_conv2d:
            assert patch_size[0] == 1
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:],
                has_bias=bias,
                pad_mode='valid',
                bias_init='zeros',
                **factory_kwargs,
            )
        else:
            self.proj = nn.Conv3d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                has_bias=bias,
                pad_mode='valid',
                bias_init='zeros',
                **factory_kwargs,
            )
        # nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        # nn.init.zeros_(self.proj.bias)
        w = self.proj.weight
        w_flatted = w.reshape(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def construct(self, x):
        # x: (B C T H W)
        if self.use_conv2d:
            B, C, T, H, W = x.shape
            # (B C T H W) -> (B*T C H W)
            x = x.permute(0, 2, 1, 3, 4).reshape((B * T, C, H, W))

        x = self.proj(x)  # (BT C' H' W')

        if self.use_conv2d:
            _, Co, Ho, Wo = x.shape
            # (B*T C H W) -> (B C T H W)
            x = x.reshape(B, T, Co, Ho, Wo).permute(0, 2, 1 ,3, 4)

        if self.flatten:
            # (B C T H W) -> (B C THW) -> (B THW C)
            x = x.flatten(start_dim=2).transpose((0, 2, 1))  # BCHW -> BNC
        x = self.norm(x)
        return x

class TextProjection(nn.Cell):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.linear_1 = mint.nn.Linear(
            in_channels,
            hidden_size,
            bias=True,
        )
        self.act_1 = act_layer()
        self.linear_2 = mint.nn.Linear(
            hidden_size,
            hidden_size,
            bias=True,
        )

    def construct(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class SinusoidalEmbedding(nn.Cell):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        half = dim // 2
        self._freqs = ms.Tensor(
            np.expand_dims(
                np.exp(-math.log(max_period) * np.arange(start=0, stop=half, dtype=np.float32) / half), axis=0
            )
        )
        self._dim = dim

    def construct(self, t):
        # AMP: cos, sin fp32
        args = t[:, None].float() * self._freqs
        embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
        if self._dim % 2:
            embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
        return embedding


def init_normal(param, mean=0., std=1.) -> None:
    param.set_data(initializer(Normal(std, mean), param.shape, param.dtype))


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.SequentialCell(
            mint.nn.Linear(
                frequency_embedding_size, hidden_size, bias=True,
            ),
            act_layer(),
            mint.nn.Linear(hidden_size, out_size, bias=True),
        )
        init_normal(self.mlp[0].weight, std=0.02)
        init_normal(self.mlp[2].weight, std=0.02)

        self.timestep_embedding = SinusoidalEmbedding(frequency_embedding_size, max_period=max_period)
        self.dtype = dtype

    def construct(self, t):
        t_freq = self.timestep_embedding(t).to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
