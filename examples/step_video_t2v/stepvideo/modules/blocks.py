# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, mint

import numpy as np
from typing import Optional
from stepvideo.modules.rope import RoPE3D
from stepvideo.modules.attentions import Attention
from stepvideo.modules.normalization import RMSNorm


class SelfAttention(Attention):
    def __init__(self, hidden_dim, head_dim, bias=False, with_rope=True, with_qk_norm=True, attn_type='mindspore', sp_group: str = None):
        self.head_dim = head_dim
        self.n_heads = hidden_dim // head_dim
        
        super().__init__(sp_group=sp_group, head_dim=self.head_dim, head_num=self.n_heads)

        self.wqkv = mint.nn.Linear(hidden_dim, hidden_dim*3, bias=bias)
        self.wo = mint.nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.with_rope = with_rope
        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=True)
        
        if self.with_rope:
            self.rope_3d = RoPE3D(freq=1e4, F0=1.0, scaling_factor=1.0)
            self.rope_ch_split = [64, 32, 32]
        
        self.core_attention = self.attn_processor(attn_type=attn_type)
        self.parallel = attn_type=='parallel'
        
    def apply_rope3d(self, x, fhw_positions, rope_ch_split, parallel=True):
        x = self.rope_3d(x, fhw_positions, rope_ch_split, parallel)
        return x
        
    def construct(
        self, 
        x,
        cu_seqlens=None,
        max_seqlen=None,
        rope_positions=None,
        attn_mask=None
    ):
        xqkv = self.wqkv(x) 
        xqkv = xqkv.view(*x.shape[:-1], self.n_heads, 3*self.head_dim)

        xq, xk, xv = mint.split(xqkv, [self.head_dim]*3, dim=-1)  ## seq_len, n, dim
    
        if self.with_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)
    
        if self.with_rope:
            xq = self.apply_rope3d(xq, rope_positions, self.rope_ch_split, parallel=self.parallel)
            xk = self.apply_rope3d(xk, rope_positions, self.rope_ch_split, parallel=self.parallel)
            
        output = self.core_attention(
                    xq,
                    xk,
                    xv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    attn_mask=attn_mask
                )
        
        # output = rearrange(output, 'b s h d -> b s (h d)')
        b, s, h, d = output.shape
        output = output.view(b, s, h*d)

        output = self.wo(output)
        
        return output
    
    
class CrossAttention(Attention):
    def __init__(self, hidden_dim, head_dim, bias=False, with_qk_norm=True, attn_type='mindspore', sp_group: str = None):
        self.head_dim = head_dim
        self.n_heads = hidden_dim // head_dim
        
        super().__init__(sp_group=sp_group, head_dim=self.head_dim, head_num=self.n_heads)

        self.wq = mint.nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.wkv = mint.nn.Linear(hidden_dim, hidden_dim*2, bias=bias)
        self.wo = mint.nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.with_qk_norm = with_qk_norm
        if self.with_qk_norm:
            self.q_norm = RMSNorm(head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(head_dim, elementwise_affine=True)
        
        self.core_attention = self.attn_processor(attn_type=attn_type)

    def construct(
            self, 
            x: Tensor,
            encoder_hidden_states: Tensor,
            attn_mask=None
        ):
        xq = self.wq(x) 
        xq = xq.view(*xq.shape[:-1], self.n_heads, self.head_dim)
        
        xkv = self.wkv(encoder_hidden_states)
        xkv = xkv.view(*xkv.shape[:-1], self.n_heads, 2*self.head_dim)

        xk, xv = mint.split(xkv, [self.head_dim]*2, dim=-1)  ## seq_len, n, dim
    
        if self.with_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        output = self.core_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=attn_mask
                )
        
        # output = rearrange(output, 'b s h d -> b s (h d)')
        b, s, h, d = output.shape
        output = output.view(b, s, h*d)

        output = self.wo(output)
        
        return output

    
class GELU(nn.Cell):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = mint.nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: Tensor) -> Tensor:
        return mint.nn.functional.gelu(gate, approximate=self.approximate)

    def construct(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states
    
    
class FeedForward(nn.Cell):
    def __init__(
        self, 
        dim: int,
        inner_dim: Optional[int] = None,
        dim_out: Optional[int] = None,
        mult: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = dim*mult if inner_dim is None else inner_dim
        dim_out = dim if dim_out is None else dim_out
        self.net = nn.CellList([
            GELU(dim, inner_dim, approximate="tanh", bias=bias),
            mint.nn.Identity(),
            mint.nn.Linear(inner_dim, dim_out, bias=bias)
        ])

    def construct(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
    

def modulate(x, scale, shift):
    x = x * (1 + scale) + shift
    return x

def gate(x, gate):
    x = gate * x
    return x


class StepVideoTransformerBlock(nn.Cell):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        attention_head_dim: int,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = False,
        attention_type: str = 'parallel',
        sp_group: str = None,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = mint.nn.LayerNorm([dim], eps=norm_eps)
        self.attn1 = SelfAttention(dim, attention_head_dim, bias=False, with_rope=True, with_qk_norm=True, attn_type=attention_type, sp_group=sp_group)
        
        self.norm2 = mint.nn.LayerNorm([dim], eps=norm_eps)
        self.attn2 = CrossAttention(dim, attention_head_dim, bias=False, with_qk_norm=True, attn_type='mindspore', sp_group=sp_group)

        self.ff = FeedForward(dim=dim, inner_dim=ff_inner_dim, dim_out=dim, bias=ff_bias)

        self.scale_shift_table = Parameter(np.random.randn(6, dim) /dim**0.5)

    def construct(
        self,
        q: Tensor,
        kv: Optional[Tensor] = None,
        timestep: Optional[Tensor] =  None,
        attn_mask = None,
        rope_positions: list = None, 
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            Tensor(chunk) for chunk in (self.scale_shift_table[None] + timestep.reshape(-1, 6, self.dim)).chunk(6, axis=1)
        )

        scale_shift_q = modulate(self.norm1(q), scale_msa, shift_msa)

        attn_q = self.attn1(
            scale_shift_q,
            rope_positions=rope_positions
        )

        q = gate(attn_q, gate_msa) + q
        
        attn_q = self.attn2(
                q,
                kv,
                attn_mask
            )

        q = attn_q + q

        scale_shift_q = modulate(self.norm2(q), scale_mlp, shift_mlp)

        ff_output = self.ff(scale_shift_q)
        
        q = gate(ff_output, gate_mlp) + q

        return ops.stop_gradient(q)
    
    
class PatchEmbed(nn.Cell):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=64,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
    ):
        super().__init__()

        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = mint.nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias,
        )

    def construct(self, latent):
        latent = self.proj(latent).to(latent.dtype)   
        if self.flatten:
            latent = mint.swapaxes(latent.flatten(start_dim=2), 1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        return latent
