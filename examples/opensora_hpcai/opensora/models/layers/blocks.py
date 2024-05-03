import math
import numbers
from typing import Optional, Tuple, Type

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import initializer

from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention
from mindone.models.modules.pos_embed import _get_1d_sincos_pos_embed_from_grid, _get_2d_sincos_pos_embed_from_grid


class Attention(nn.Cell):
    def __init__(self, dim_head: int, attn_drop: float = 0.0, attn_dtype=ms.float32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.attn_dtype = attn_dtype

    def construct(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        args:
            q: (b n_q h d), h - num_head, n_q - seq_len of q
            k v: (b n_k h d), (b h n_v d)
            mask: (b 1 n_k), 0 - keep, 1 indicates discard.
        return:
            ms.Tensor (b n_q h d)
        """

        # (b n h d) -> (b h n d)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        b, h, n_q, d = q.shape
        _, _, n_k, _ = k.shape

        q = ops.reshape(q, (b * h, n_q, d))
        k = ops.reshape(k, (b * h, n_k, d))
        v = ops.reshape(v, (b * h, n_k, d))

        q = q.to(self.attn_dtype)
        k = k.to(self.attn_dtype)
        v = v.to(self.attn_dtype)

        sim = ops.matmul(q, k.transpose(0, 2, 1)) * self.scale

        sim = sim.to(ms.float32)  # (b h n_q n_k)

        if mask is not None:
            # (b 1 n_k) -> (b*h 1 n_k)
            mask = ops.repeat_interleave(mask, h, axis=0)
            mask = mask.to(ms.bool_)
            sim = ops.masked_fill(sim, mask, -ms.numpy.inf)

        # (b h n_q n_k)
        attn = ops.softmax(sim, axis=-1).astype(v.dtype)
        attn = self.attn_drop(attn)
        out = ops.matmul(attn, v)

        out = ops.reshape(out, (b, h, -1, d))
        # (b h n d) -> (b n h d)
        out = ops.transpose(out, (0, 2, 1, 3))
        return out


class MultiHeadCrossAttention(nn.Cell):
    """
    This implementation is more friendly to mindspore in graph mode currently.
    Overhead computation lies in the padded tokens in a batch, which is padded
    to a fixed length max_tokens. If the prompts are short, this overhead can be high.

    TODO: remove the computation on the padded sequence, referring to xformers, or
    reduce it by padding to the max prompt length in the batch instead of a fixed large value.
        Here is how torch support dynamic text length in a batch. diagnonal maksing for valid texts. more memory efficient for short prompts.
        ```
        attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        ```
    """

    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, has_bias=True, enable_flash_attention=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # TODO: model impr: remove bias
        self.q_linear = nn.Dense(d_model, d_model, has_bias=has_bias)
        self.kv_linear = nn.Dense(d_model, d_model * 2, has_bias=has_bias)

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )
        if self.enable_flash_attention:
            attn_dtype = ms.bfloat16
            assert attn_drop == 0.0, "attn drop is not supported in FA currently."
            self.flash_attention = MSFlashAttention(
                head_dim=self.head_dim,
                head_num=self.num_heads,
                attention_dropout=attn_drop,
                input_layout="BSH",
                dtype=attn_dtype,
            )
        else:
            # TODO: test ms.bfloat16 for vanilla attention
            attn_dtype = ms.float32
            self.attention = Attention(self.head_dim, attn_drop=attn_drop, attn_dtype=attn_dtype)

        self.proj = nn.Dense(d_model, d_model, has_bias=has_bias).to_float(attn_dtype)
        self.proj_drop = nn.Dropout(p=proj_drop).to_float(attn_dtype)

    def construct(self, x, cond, mask=None):
        """
        Inputs:
            x: (B, N, C), N=seq_len=h*w*t, C = hidden_size = head_dim * num_heads
            cond: (1, B*N_tokens, C)
            mask : (B, N_tokens), 1 - valid tokens, 0 - padding tokens
        Return:
            (B, N, C)
        """
        B, N, C = x.shape

        # cond: (1, B*N_tokens, C) -> (B, N_tokens, C)
        cond = ops.reshape(cond, (B, -1, C))
        N_k = cond.shape[1]

        # 1. q, kv linear projection
        q = self.q_linear(x)  # .reshape((1, -1, self.num_heads, self.head_dim))
        kv = self.kv_linear(cond)  # .reshape((1, -1, 2, self.num_heads, self.head_dim))

        # 2. reshape qkv for multi-head attn
        # q: (B N C) -> (B N num_head head_dim)
        q = ops.reshape(q, (B, N, self.num_heads, self.head_dim))

        # kv: (B N_k C*2) -> (B N_k 2 C) -> (B N_k 2 num_head head_dim).
        kv = ops.reshape(kv, (B, N_k, 2, self.num_heads, self.head_dim))
        k, v = ops.split(kv, 1, axis=2)
        # (b n h d)
        k = ops.squeeze(k, axis=2)
        v = ops.squeeze(v, axis=2)

        # 2+: mask adaptation for multi-head attention
        if mask is not None:
            # flip mask, since ms FA treats 1 as discard, 0 as retain.
            mask = 1 - mask

        # 3. attn compute
        if self.enable_flash_attention:
            if mask is not None:
                # (b n_k) -> (b 1 1 n_k), will be broadcast according to qk sim, e.g. (b num_heads n_q n_k)
                mask = mask[:, None, None, :]
                # (b 1 1 n_k) -> (b 1 n_q n_k)
                # mask = ops.repeat_interleave(mask.to(ms.uint8), q.shape[-2], axis=-2)
                mask = ops.repeat_interleave(mask, int(q.shape[1]), axis=-2)
            x = self.flash_attention(q, k, v, mask=mask)

            # FA attn_mask def: retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, `(B, 1, S1, S2)` `(S1, S2)`
        else:
            if mask is not None:
                mask = mask[:, None, :]
            x = self.attention(q, k, v, mask)

        x = ops.reshape(x, (B, N, -1))

        # 4. output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SelfAttention(nn.Cell):
    """Attention adopted from :
    Multi-head self attention
    https://github.com/pprp/timm/blob/master/timm/models/vision_transformer.py
    Args:
        dim (int): hidden size.
        num_heads (int): number of heads
        qkv_bias (int): whether to use bias
        attn_drop (bool): attention dropout
        proj_drop (bool): projection dropout
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        enable_flash_attention=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias, weight_init="XavierUniform", bias_init="Zero")

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            attn_dtype = ms.bfloat16
            self.flash_attention = MSFlashAttention(
                head_dim=head_dim,
                head_num=num_heads,
                attention_dropout=attn_drop,
                input_layout="BSH",
                dtype=attn_dtype,
            )
        else:
            # TODO: support ms.bfloat16
            attn_dtype = ms.float32
            self.attention = Attention(head_dim, attn_drop=attn_drop, attn_dtype=attn_dtype)

        self.proj = nn.Dense(dim, dim, weight_init="XavierUniform", bias_init="Zero").to_float(attn_dtype)
        self.proj_drop = nn.Dropout(p=proj_drop).to_float(attn_dtype)

    def construct(self, x, mask=None):
        """
        x: (b n c)
        mask: (b n), 1 - valid, 0 - padded
        """
        x_dtype = x.dtype
        B, N, C = x.shape

        qkv = self.qkv(x)
        # (b, n, 3*h*d) -> (b, n, 3, h, d)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        q, k, v = ops.split(qkv, 1, axis=2)  # (b n h d)
        q = ops.squeeze(q, axis=2)
        k = ops.squeeze(k, axis=2)
        v = ops.squeeze(v, axis=2)

        # mask process
        if mask is not None:
            mask = 1 - mask

        if self.enable_flash_attention:
            if mask is not None:
                mask = mask[:, None, None, :]
                # mask: (b n_k) -> (b 1 n_q n_k)
                mask = ops.repeat_interleave(mask, int(q.shape[1]), axis=-2)
            out = self.flash_attention(q, k, v, mask=mask)
        else:
            if mask is not None:
                mask = mask[:, None, :]
            out = self.attention(q, k, v, mask)

        # reshape FA output to original attn input format (b n h*d)
        out = out.view(B, N, -1)

        return self.proj_drop(self.proj(out)).to(x_dtype)


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            self.beta = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
        else:
            self.gamma = ops.ones(normalized_shape, dtype=dtype)
            self.beta = ops.zeros(normalized_shape, dtype=dtype)
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: Tensor):
        oridtype = x.dtype
        x, _, _ = self.layer_norm(x.to(ms.float32), self.gamma.to(ms.float32), self.beta.to(ms.float32))
        return x.to(oridtype)


class GELU(nn.GELU):
    def __init__(self, approximate: str = "none"):
        if approximate == "none":
            super().__init__(False)
        elif approximate == "tanh":
            super().__init__(True)
        else:
            raise ValueError(f"approximate must be one of ['none', 'tanh'], but got {approximate}.")


approx_gelu = lambda: GELU(approximate="tanh")


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PatchEmbed3D(nn.Cell):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode="valid", has_bias=True
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def construct(self, x):
        # padding
        _, _, D, H, W = x.shape
        if W % self.patch_size[2] != 0:
            x = ops.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = ops.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = ops.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.shape[2], x.shape[3], x.shape[4]
            x = x.flatten(start_dim=2).swapaxes(1, 2)
            x = self.norm(x)
            x = x.swapaxes(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(start_dim=2).swapaxes(1, 2)  # BCTHW -> BNC
        return x


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.SequentialCell(
            nn.Dense(frequency_embedding_size, hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(hidden_size, hidden_size, has_bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = ops.exp(-math.log(max_period) * ops.arange(start=0, end=half, dtype=ms.float32) / half)
        args = t[:, None].float() * freqs[None]
        embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
        if dim % 2:
            embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def construct(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # diff
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class T2IFinalLayer(nn.Cell):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # (1152, 4*8)
        self.linear = nn.Dense(hidden_size, num_patch * out_channels, has_bias=True)
        self.scale_shift_table = ms.Parameter(ops.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def construct(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, axis=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CaptionEmbedder(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    # FIXME: rm nn.GELU instantiate for parallel training
    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU, token_num=120):
        super().__init__()

        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )

        y_embedding = ops.randn(token_num, in_channels) / in_channels**0.5
        # just for token dropping replacement, not learnable
        self.y_embedding = ms.Parameter(Tensor(y_embedding, dtype=ms.float32), requires_grad=False)

        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = ops.rand(caption.shape[0]) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1

        # manually expand dims to avoid infer-shape bug in ms2.3 daily
        caption = ops.where(
            drop_ids[:, None, None, None], self.y_embedding[None, None, :, :], caption.to(self.y_embedding.dtype)
        )

        return caption

    def construct(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding

    Args:
        image_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(
        self,
        image_size: Optional[int] = 224,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 96,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size: Tuple = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        if image_size is not None:
            self.image_size: Optional[Tuple] = (image_size, image_size) if isinstance(image_size, int) else image_size
            self.patches_resolution: Optional[Tuple] = tuple([s // p for s, p in zip(self.image_size, self.patch_size)])
            self.num_patches: Optional[int] = self.patches_resolution[0] * self.patches_resolution[1]
        else:
            self.image_size: Optional[Tuple] = None
            self.patches_resolution: Optional[Tuple] = None
            self.num_patches: Optional[int] = None
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode="pad", has_bias=bias
        )

    def construct(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if self.image_size is not None:
            assert (h, w) == (
                self.image_size[0],
                self.image_size[1],
            ), f"Input height and width ({h},{w}) doesn't match model ({self.image_size[0]},{self.image_size[1]})."
        x = self.proj(x)
        x = ops.reshape(x, (b, self.embed_dim, -1))
        x = ops.transpose(x, (0, 2, 1))  # B Ph*Pw C
        return x


class LinearPatchEmbed(nn.Cell):
    """Image to Patch Embedding: using a linear layer instead of conv2d layer for projection

    Args:
        image_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(
        self,
        image_size: Optional[int] = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size: Tuple = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        if image_size is not None:
            self.image_size: Optional[Tuple] = (image_size, image_size) if isinstance(image_size, int) else image_size
            self.patches_resolution: Optional[Tuple] = tuple([s // p for s, p in zip(self.image_size, self.patch_size)])
            self.num_patches: Optional[int] = self.patches_resolution[0] * self.patches_resolution[1]
        else:
            self.image_size: Optional[Tuple] = None
            self.patches_resolution: Optional[Tuple] = None
            self.num_patches: Optional[int] = None
        self.embed_dim = embed_dim
        self.proj = nn.Dense(patch_size * patch_size * in_chans, embed_dim, has_bias=bias)

    def construct(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if self.image_size is not None:
            assert (h, w) == (
                self.image_size[0],
                self.image_size[1],
            ), f"Input height and width ({h},{w}) doesn't match model ({self.image_size[0]},{self.image_size[1]})."
        ph, pw = h // self.patch_size[0], w // self.patch_size[1]
        x = x.reshape((b, c, self.patch_size[0], ph, self.patch_size[1], pw))  # (B, C, P, Ph, P, Pw)
        # x = x.transpose((0, 3, 5, 2, 4, 1))  # (B, Ph, Pw, P, P, C)
        x = x.transpose((0, 3, 5, 2, 4, 1))  # (B, Ph, Pw, P, P, C)
        x = x.reshape((b, ph * pw, self.patch_size[0] * self.patch_size[1] * c))  # (B, Ph*Pw, P*P*C)

        x = self.proj(x)  # B Ph*Pw C_out
        return x


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Cell] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class LabelEmbedder(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = ops.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = ops.where(drop_ids, self.num_classes, labels)
        return labels

    def construct(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
