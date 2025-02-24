# -*- coding: utf-8 -*-
import math

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import ops, nn, Tensor


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def gelu(x):
    return 0.5 * x * (1.0 + ops.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


@ms.jit
def gelu_jit(x):
    """OpenAI's gelu implementation."""
    return gelu(x)


class GELUJit(nn.Cell):
    def construct(self, input: Tensor) -> Tensor:
        return gelu_jit(input)


def get_activation(activation):
    if activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu_jit':
        return GELUJit()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'unknown activation type {activation}')


# todo: wtf dtype?
class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, dtype=None):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)

    def construct(self, x):
        x_shape = x.shape
        if x.ndim >= 3:
            x = x.view(x_shape[0], x_shape[1], x_shape[2], -1)
        y = super().construct(x).to(x.dtype)
        return y.view(x_shape)


class AttentionPooling(nn.Cell):

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = ms.Parameter(ops.randn(1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def construct(self, x):
        bs, length, width = x.shape

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose((0, 2, 1, 3))
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs*self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose((0, 2, 1))
            return x

        class_token = x.mean(axis=1, keep_dims=True) + self.positional_embedding.to(x.dtype)
        x = ops.cat([class_token, x], axis=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        assert q.dtype == k.dtype == v.dtype
        ori_dtype = q.dtype  # BatchMatMul has to run in fp16 mode on ascend!
        q, k, v = q.to(ms.float16), k.to(ms.float16), v.to(ms.float16)
        weight = ops.BatchMatMul(transpose_a=True)(  # 'bct,bcs->bts',
            q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = ops.softmax(weight.float(), axis=-1).to(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = ops.BatchMatMul(transpose_b=True)(v, weight)  # 'bcs,bts->bct'
        a = a.to(ori_dtype)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose((0, 2, 1))

        return a[:, 0, :]  # cls_token


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Dense(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.get_parameters():
        p.set_data(init.initializer("zeros", p.shape, p.dtype))
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.get_parameters():
        p.set_data(init.initializer(scale * p.data))
    return module


def normalization(channels, dtype=None):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Cell for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=32, dtype=dtype)


def timestep_embedding(timesteps, dim, max_period=10000, dtype=None):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if dtype is None:
        dtype = ms.float32
    half = dim // 2
    freqs = ops.exp(
        -math.log(max_period) * ops.arange(start=0, end=half, dtype=ms.float32) / half
    ).to(dtype=dtype)
    args = timesteps[:, None].to(dtype) * freqs[None]
    embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
    if dim % 2:
        embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def attention(q, k, v, d_k):
    scores = ops.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = ops.softmax(scores, axis=-1)
    output = ops.matmul(scores, v)
    return output
