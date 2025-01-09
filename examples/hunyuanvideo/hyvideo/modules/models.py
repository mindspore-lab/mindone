from typing import Any, List, Tuple, Optional, Union, Dict
import math
# from einops import rearrange

import mindspore as ms
from mindspore import nn, ops
import mindspore.ops.functional as F

from .norm_layers import LayerNorm, get_norm_layer
from .activation_layers import get_activation_layer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .mlp_layers import MLP # , MLPEmbedder, FinalLayer
from .posemb_layers import apply_rotary_emb
from .attention import attention, VanillaAttention #, parallel_attention, get_cu_seqlens

'''
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .token_refiner import SingleTokenRefiner
'''

def rearrange_qkv(qkv, heads_num):
    # qkv: shape (B L K*H*D), K=3
    # B L (K H D) -> B L K H D -> K B L H D
    # return q/k/v: (B L H D)
    B, L, KHD = qkv.shape
    H = heads_num
    # D = head_dim # KHD // (K * self.heads_num)
    D = KHD // (3 * H)
    qkv = ops.reshape(qkv, (B, L, 3, H, D))
    q, k, v = ops.split(qkv, 1, axis=2)
    q = ops.squeeze(q, axis=2)
    k = ops.squeeze(k, axis=2)
    v = ops.squeeze(v, axis=2)

    return q, k, v


class MMDoubleStreamBlock(nn.Cell):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_attn_qkv = nn.Dense(
            hidden_size, hidden_size * 3, has_bias=qkv_bias)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Dense(
            hidden_size, hidden_size, has_bias=qkv_bias)

        self.img_norm2 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Dense(
            hidden_size, hidden_size * 3, has_bias=qkv_bias
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Dense(
            hidden_size, hidden_size, has_bias=qkv_bias,
        )

        self.txt_norm2 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        #
        self.compute_attention = VanillaAttention(head_dim)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def construct(
        self,
        img: ms.Tensor,
        txt: ms.Tensor,
        vec: ms.Tensor,
        cu_seqlens_q: Optional[ms.Tensor] = None,
        cu_seqlens_kv: Optional[ms.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
    ):
        # img:
        # txt:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, axis=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, axis=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        # "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        img_q, img_k, img_v = rearrange_qkv(img_qkv, self.heads_num)

        # Apply QK-Norm if needed
        # TODO: check whether need to cast to dtype of img_v
        img_q = self.img_attn_q_norm(img_q) # .to(img_v)
        img_k = self.img_attn_k_norm(img_k) # .to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            # assert (
            #    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            # ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)

        # "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        txt_q, txt_k, txt_v = rearrange_qkv(txt_qkv, self.heads_num)

        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q) # .to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k) # .to(txt_v)

        # Run actual attention.
        q = ops.concat((img_q, txt_q), axis=1)
        k = ops.concat((img_k, txt_k), axis=1)
        v = ops.concat((img_v, txt_v), axis=1)
        # assert (
        #    cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        # ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        # attention computation start

        attn = self.compute_attention(q, k, v)
        # TODO: support FA and parallel attn

        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt

class MMSingleStreamBlock(nn.Cell):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Dense(
            hidden_size, hidden_size * 3 + mlp_hidden_dim,
        )
        # proj and mlp_out
        self.linear2 = nn.Dense(
            hidden_size + mlp_hidden_dim, hidden_size,
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        self.compute_attention = VanillaAttention(head_dim)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def construct(
        self,
        x: ms.Tensor,
        vec: ms.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[ms.Tensor] = None,
        cu_seqlens_kv: Optional[ms.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[ms.Tensor, ms.Tensor] = None,
    ) -> ms.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, axis=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = ops.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], axis=-1
        )

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q, k, v = rearrange_qkv(qkv, heads_num=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q) # .to(v)
        k = self.k_norm(k) # .to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            # assert (
            #    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            # ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = ops.concat((img_q, txt_q), axis=1)
            k = ops.concat((img_k, txt_k), axis=1)

        # Compute attention.
        # assert (
        #     cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        # ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        # attention computation start
        attn = self.compute_attention(
            q,
            k,
            v,
        )
        # TODO: add FA
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(ops.concat((attn, self.mlp_act(mlp)), axis=2))
        return x + apply_gate(output, gate=mod_gate)


