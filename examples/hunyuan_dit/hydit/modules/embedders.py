import math

import mindspore as ms
from mindspore import nn, ops

MAX_PERIOD = -float(math.log(10000))


class PatchEmbed(nn.Cell):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            img_size = tuple(img_size)
        else:
            raise ValueError(f"img_size must be int or tuple/list of length 2. Got {img_size}")
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def update_image_size(self, img_size):
        self.img_size = img_size
        self.grid_size = (img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def construct(self, x):
        # B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(start_dim=2).swapaxes(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def timestep_embedding(t, dim, max_period=MAX_PERIOD, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    if not repeat_only:
        half = dim // 2
        freqs = ops.exp(
            (max_period * ops.arange(start=0, end=half, dtype=ms.float32) / half).float()
        )  # size: [dim/2], 一个指数衰减的曲线
        args = t[:, None].float() * freqs[None]
        embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
        if dim % 2:
            embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
    else:
        broadcast_shape = t.shape + (dim,)
        embedding = t.unsqueeze(-1).broadcast_to(broadcast_shape)
    return embedding


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, out_size, has_bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = -float(math.log(10000))

    def construct(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
