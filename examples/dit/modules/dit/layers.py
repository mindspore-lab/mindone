"""
DiT Blocks and Layers
"""
import collections.abc
import logging
from functools import partial
from itertools import repeat

import numpy as np

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import XavierUniform, Zero

from mindone.utils.version_control import check_valid_flash_attention, choose_flash_attention_dtype

logger = logging.getLogger(__name__)
FLASH_IS_AVAILABLE = check_valid_flash_attention()
if FLASH_IS_AVAILABLE:
    from mindspore.nn.layer.flash_attention import FlashAttention

    logger.info("Flash attention is available.")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class Mlp(nn.Cell):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks, adopted from:
    https://github.com/pprp/timm/blob/master/timm/layers/mlp.py
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU(approximate=False),
        has_bias=True,
        drop=0.0,
        use_conv=False,
        dtype=ms.float32,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(has_bias)
        drop_probs = to_2tuple(drop)
        self.dtype = dtype
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Dense

        self.fc1 = linear_layer(
            in_features, hidden_features, has_bias=bias[0], weight_init=XavierUniform(), bias_init=Zero()
        ).to_float(self.dtype)
        self.act = act_layer
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.fc2 = linear_layer(
            hidden_features, out_features, has_bias=bias[1], weight_init=XavierUniform(), bias_init=Zero()
        ).to_float(self.dtype)
        self.drop2 = nn.Dropout(p=drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def exists(val):
    return val is not None


class Attention(nn.Cell):
    def __init__(self, dim_head, attn_drop=0.0):
        super().__init__()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.scale = dim_head**-0.5
        self.attn_drop = nn.Dropout(p=attn_drop)

    def construct(self, q, k, v, mask):
        sim = ops.matmul(q, self.transpose(k, (0, 2, 1))) * self.scale

        if exists(mask):
            mask = self.reshape(mask, (mask.shape[0], -1))
            if sim.dtype == ms.float16:
                finfo_type = np.float16
            else:
                finfo_type = np.float32
            max_neg_value = -np.finfo(finfo_type).max
            mask = mask.repeat(self.heads, axis=0)
            mask = ops.expand_dims(mask, axis=1)
            sim.masked_fill(mask, max_neg_value)

        # use fp32 for exponential inside
        attn = self.softmax(sim.astype(ms.float32)).astype(v.dtype)
        attn = self.attn_drop(attn)

        out = ops.matmul(attn, v)

        return out


class SelfAttention(nn.Cell):
    """Attention adopted from :
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
        dtype=ms.float32,
        enable_flash_attention=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dtype = dtype
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias, weight_init=XavierUniform(), bias_init=Zero()).to_float(
            self.dtype
        )
        self.proj = nn.Dense(dim, dim, weight_init=XavierUniform(), bias_init=Zero()).to_float(self.dtype)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

        self.attention = Attention(head_dim, attn_drop=attn_drop)

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            self.flash_attention = FlashAttention(
                head_dim=head_dim,
                head_num=num_heads,
                high_precision=True,
                dropout_rate=attn_drop,
            )  # TODO: how high_precision affect the training or inference quality
            self.fa_mask_dtype = choose_flash_attention_dtype()  # ms.uint8 or ms.float16 depending on version
            # logger.info("Flash attention is enabled.")
        else:
            self.flash_attention = None

    @staticmethod
    def _rearange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = ops.reshape(x, (b, n, h, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = ops.reshape(x, (b, h, n, d))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, h * d))
        return x

    def construct(self, x, mask=None):
        x_dtype = x.dtype
        h = self.num_heads
        B, N, C = x.shape
        # (b, n, 3*h*d) -> (b, n, 3, h*d)  -> (3, b, n, h*d)
        qkv = self.qkv(x).reshape(B, N, 3, -1).permute((2, 0, 1, 3))
        q, k, v = qkv.unbind(0)
        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // h

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= 256
        ):  # TODO: why restrict head_dim?
            # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            if mask is None:
                mask = ops.zeros((q_b, q_n, q_n), self.fa_mask_dtype)

            out = self.flash_attention(
                q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(self.fa_mask_dtype)
            )

            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
            out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        else:
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)

            out = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b, n, h*d)
            out = self._rearange_out(out, h)

        return self.proj_drop(self.proj(out)).to(x_dtype)


class DiTBlock(nn.Cell):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dtype=ms.float32, **block_kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.norm = ops.LayerNorm(-1, -1, epsilon=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = nn.GELU(approximate=True)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.SequentialCell(
            nn.SiLU().to_float(self.dtype),
            nn.Dense(
                hidden_size, 6 * hidden_size, has_bias=True, weight_init=XavierUniform(), bias_init=Zero()
            ).to_float(self.dtype),
        )

    def construct(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, axis=1)
        ones = ops.ones((self.hidden_size))
        zeros = ops.zeros((self.hidden_size))
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm(x, ones, zeros)[0], shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm(x, ones, zeros)[0], shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Cell):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.norm_final = ops.LayerNorm(-1, -1, epsilon=1e-6)
        self.linear = nn.Dense(
            hidden_size,
            patch_size * patch_size * out_channels,
            has_bias=True,
            weight_init=XavierUniform(),
            bias_init=Zero(),
        ).to_float(self.dtype)

        self.adaLN_modulation = nn.SequentialCell(
            nn.SiLU().to_float(self.dtype),
            nn.Dense(
                hidden_size, 2 * hidden_size, has_bias=True, weight_init=XavierUniform(), bias_init=Zero()
            ).to_float(self.dtype),
        )

    def construct(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        ones = ops.ones((self.hidden_size))
        zeros = ops.zeros((self.hidden_size))
        x = modulate(self.norm_final(x, ones, zeros)[0], shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=ms.float32):
        super().__init__()
        self.dtype = dtype
        self.mlp = nn.SequentialCell(
            nn.Dense(
                frequency_embedding_size, hidden_size, has_bias=True, weight_init=XavierUniform(), bias_init=Zero()
            ).to_float(self.dtype),
            nn.SiLU().to_float(self.dtype),
            nn.Dense(hidden_size, hidden_size, has_bias=True, weight_init=XavierUniform(), bias_init=Zero()).to_float(
                self.dtype
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2
            freqs = ops.exp(
                -ops.log(ms.Tensor(max_period, ms.float32))
                * ms.numpy.arange(start=0, stop=half, dtype=ms.float32)
                / half
            )
            args = timesteps[:, None] * freqs[None]
            embedding = ops.concat((ops.cos(args), ops.sin(args)), axis=-1)
            if dim % 2:
                embedding = ops.concat((embedding, ops.ZerosLike()(embedding[:, :1])), axis=-1)
        else:
            embedding = ops.reshape(timesteps.repeat(dim), (-1, dim))
        return embedding

    def construct(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob, dtype=ms.float32):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.dtype = dtype
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size).to_float(self.dtype)
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


class PatchEmbed(nn.Cell):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=ms.float32,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.dtype = dtype

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=bias).to_float(
            self.dtype
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm = self.norm.to_float(self.dtype)

    def construct(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = ops.flatten(x, start_dim=2).permute((0, 2, 1))  # BCHW -> BNC
        x = self.norm(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
