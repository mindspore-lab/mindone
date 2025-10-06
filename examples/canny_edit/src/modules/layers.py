import math
from dataclasses import dataclass
from typing import Union  # Import Tuple and Union from typing

from src.modules.math import apply_rope, attention, rope

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, mint, ops

from mindone.transformers.mindspore_adapter.utils import _DTYPE_2_MIN


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None, training=True
):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = attn_mask.to(ms.float32)
            attn_mask = attn_mask.masked_fill((1 - attn_mask).to(ms.bool_), _DTYPE_2_MIN[ms.float16])
        attn_mask = attn_mask.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.transpose(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_mask,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)
    else:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = mint.zeros((L, S), dtype=query.dtype)
        if is_causal:
            temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias = ops.masked_fill(attn_bias, mint.logical_not(temp_mask), _DTYPE_2_MIN[ms.float16])
            attn_bias = attn_bias.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.transpose(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_bias,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)

    attn_weight = mint.nn.functional.dropout(attn_weight, p=dropout_p, training=training)

    out = mint.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out


# change
def scaled_dot_product_attention2(
    query,
    key,
    value,
    image_size,
    dropout_p=0.0,
    is_causal=False,
    attn_mask=None,
    union_mask=None,
    local_mask_list=[],
    local_t2i_strength=1,
    context_t2i_strength=1,
    locali2i_strength=1,
    local2out_i2i_strength=1,
    num_edit_region=1,
    scale=None,
    enable_gqa=False,
):
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = mint.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = mint.ones((L, S), dtype=ms.bool).tril(diagonal=0)
        attn_bias = ops.masked_fill(attn_bias, mint.logical_not(temp_mask), _DTYPE_2_MIN[ms.float16])
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_bias = ops.masked_fill(attn_bias, mint.logical_not(attn_mask), _DTYPE_2_MIN[ms.float16])
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.shape[-3] // key.shape[-3], -3)
        value = value.repeat_interleave(query.shape[-3] // value.shape[-3], -3)

    attn_weight = mint.matmul(query, mint.transpose(key, -2, -1)) * scale_factor
    attn_weight += attn_bias

    # Attention Amplification
    # amplify the attention between the local text prompt and local edit region
    curr_atten = attn_weight[:, :, -image_size:, 512 : 512 * (num_edit_region + 1)].copy()
    attn_weight[:, :, -image_size:, 512 : 512 * (num_edit_region + 1)] = mint.where(
        union_mask == 1, curr_atten, curr_atten * (local_t2i_strength)
    )
    # amplify the attention between the target prompt and the whole image
    curr_atten1 = attn_weight[:, :, -image_size:, :512].copy()
    attn_weight[:, :, -image_size:, :512] = curr_atten1 * (context_t2i_strength)

    for local_mask in local_mask_list:
        # outside the union of masks is 1
        mask1_flat = union_mask.flatten()  # (local_mask).flatten()
        mask1_indices = 512 * (num_edit_region + 1) + mint.nonzero(mask1_flat, as_tuple=True)[0]
        # mask2_flat inside the mask is 1
        mask2_flat = (1 - local_mask).flatten()
        mask2_indices = 512 * (num_edit_region + 1) + mint.nonzero(mask2_flat, as_tuple=True)[0]
        # inside the other masks is 1
        mask3_flat = 1 - mint.logical_or(mask1_flat.bool(), mask2_flat.bool()).int()
        mask3_indices = 512 * (num_edit_region + 1) + mint.nonzero(mask3_flat, as_tuple=True)[0]

        # amplify the attention within the edit region
        attn_weight[:, :, mask2_indices[:, None], mask2_indices] = (
            locali2i_strength * attn_weight[:, :, mask2_indices[:, None], mask2_indices]
        )
        # amplify the attention between the edit region and the bg region
        attn_weight[:, :, mask2_indices[:, None], mask1_indices] = (
            local2out_i2i_strength * attn_weight[:, :, mask2_indices[:, None], mask1_indices]
        )
        # amplify the attention between the edit region and other edit regions
        attn_weight[:, :, mask2_indices[:, None], mask3_indices] = (
            local2out_i2i_strength * attn_weight[:, :, mask2_indices[:, None], mask3_indices]
        )

    # END of Amplification

    attn_weight = mint.nn.functional.softmax(attn_weight, dim=-1, dtype=ms.float32).astype(query.dtype)
    attn_weight = mint.nn.functional.dropout(attn_weight, p=dropout_p, training=True)

    out = mint.matmul(attn_weight, value)
    out = out.astype(query.dtype)
    return out


class EmbedND(nn.Cell):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def construct(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = mint.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = mint.exp(-math.log(max_period) * mint.arange(start=0, end=half, dtype=ms.float32) / half)

    args = t[:, None].float() * freqs[None]
    embedding = mint.cat([mint.cos(args), mint.sin(args)], dim=-1)
    if dim % 2:
        embedding = mint.cat([embedding, mint.zeros_like(embedding[:, :1])], dim=-1)
    if t.dtype in [ms.float16, ms.float32, ms.float64, ms.bfloat16]:
        embedding = embedding.to(t.dtype)
    return embedding


class MLPEmbedder(nn.Cell):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = mint.nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = mint.nn.SiLU()
        self.out_layer = mint.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def construct(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Cell):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = ms.Parameter(mint.ones(dim))

    def construct(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = mint.rsqrt(mint.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Cell):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def construct(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v.dtype), k.to(v.dtype)


class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        print("2" * 30)

        qkv = attn.qkv(x)
        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, L, 3, self.num_heads, -1)
        q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x


class SelfAttention(nn.Cell):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = mint.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = mint.nn.Linear(dim, dim)

    def construct(self, x: Tensor):
        # a dummy construct function to avoid error
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Cell):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = mint.nn.Linear(dim, self.multiplier * dim, bias=True)

    def construct(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(mint.nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        B, L, _ = img_qkv.shape
        img_qkv_reshaped = img_qkv.reshape(B, L, 3, attn.num_heads, attn.head_dim)
        img_q, img_k, img_v = img_qkv_reshaped.permute(2, 0, 3, 1, 4)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        txt_qkv = attn.txt_attn.qkv(txt_modulated)

        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        B, L, _ = txt_qkv.shape
        txt_qkv_reshaped = txt_qkv.reshape(B, L, 3, attn.num_heads, attn.head_dim)
        txt_q, txt_k, txt_v = txt_qkv_reshaped.permute(2, 0, 3, 1, 4)

        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # change
        if "regional_attention_mask" in attention_kwargs:
            q = mint.cat((txt_q, img_q), dim=2)
            k = mint.cat((txt_k, img_k), dim=2)
            v = mint.cat((txt_v, img_v), dim=2)
            q, k = apply_rope(q, k, pe)
            attention_mask = attention_kwargs["regional_attention_mask"]
            if "union_mask" in attention_kwargs:
                x = scaled_dot_product_attention2(
                    q,
                    k,
                    v,
                    attention_kwargs["image_size"],
                    dropout_p=0.0,
                    is_causal=False,
                    attn_mask=attention_mask,
                    union_mask=attention_kwargs["union_mask"],
                    local_mask_list=attention_kwargs["local_mask_all_dilate"],
                    local_t2i_strength=attention_kwargs["local_t2i_strength"],
                    context_t2i_strength=attention_kwargs["context_t2i_strength"],
                    locali2i_strength=attention_kwargs["local_i2i_strength"],
                    local2out_i2i_strength=attention_kwargs["local2out_i2i_strength"],
                    num_edit_region=attention_kwargs["num_edit_region"],
                )

            else:
                x = scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)

            # attn1 = rearrange(x, "B H L D -> B L (H D)")
            B, H, L, D = x.shape
            attn1 = x.permute(0, 2, 1, 3).reshape(B, L, -1)
            txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]
        else:
            q = mint.cat((txt_q, img_q), dim=2)
            k = mint.cat((txt_k, img_k), dim=2)
            v = mint.cat((txt_v, img_v), dim=2)
            attn1 = attention(q, k, v, pe=pe)
            txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        return img, txt


class DoubleStreamBlock(nn.Cell):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.SequentialCell(
            mint.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = ms.nn.SequentialCell(
            mint.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def construct(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float = 1.0,
        attention_kwargs={},
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            return self.processor(self, img, txt, vec, pe, attention_kwargs)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Cell, x: Tensor, vec: Tensor, pe: Tensor, attention_kwargs) -> Tensor:
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = mint.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        B, L, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, L, 3, attn.num_heads, -1)
        q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4)
        q, k = attn.norm(q, k, v)

        # change
        if "regional_attention_mask" in attention_kwargs:
            q, k = apply_rope(q, k, pe)
            attention_mask = attention_kwargs["regional_attention_mask"]

            if "union_mask" in attention_kwargs:
                attn_1 = scaled_dot_product_attention2(
                    q,
                    k,
                    v,
                    attention_kwargs["image_size"],
                    dropout_p=0.0,
                    is_causal=False,
                    attn_mask=attention_mask,
                    union_mask=attention_kwargs["union_mask"],
                    local_mask_list=attention_kwargs["local_mask_all_dilate"],
                    local_t2i_strength=attention_kwargs["local_t2i_strength"],
                    context_t2i_strength=attention_kwargs["context_t2i_strength"],
                    locali2i_strength=attention_kwargs["local_i2i_strength"],
                    local2out_i2i_strength=attention_kwargs["local2out_i2i_strength"],
                    num_edit_region=attention_kwargs["num_edit_region"],
                )
            else:
                attn_1 = scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)

            # attn_1 = rearrange(attn_1, "B H L D -> B L (H D)")
            B, H, L, D = attn_1.shape
            attn_1 = attn_1.permute(0, 2, 1, 3).reshape(B, L, -1)
        else:
            attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(mint.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output

        return output


class SingleStreamBlock(nn.Cell):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: Union[float, None] = None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = mint.nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = mint.nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = mint.nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def construct(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Union[Tensor, None] = None,
        ip_scale: float = 1.0,
        attention_kwargs={},
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe, attention_kwargs)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)


class LastLayer(nn.Cell):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = mint.nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = ms.nn.SequentialCell(
            mint.nn.SiLU(), mint.nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def construct(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
