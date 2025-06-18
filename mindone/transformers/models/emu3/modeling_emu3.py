# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/huggingface/transformers/blob/52daf4ec768fb9ffe84a0c373834172a7c54aecc/src/transformers/models/llama/modeling_llama.py
#
""" MindSpore Emu3 model."""
import math
from functools import cached_property
from typing import List, Optional, Tuple, Union

from transformers import Emu3Config, Emu3TextConfig, Emu3VQVAEConfig
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import mint, nn, ops, jit_class, _no_grad
from mindspore.common.initializer import (
    Constant,
    HeNormal,
    Normal,
    Uniform,
    initializer,
)
from mindspore.communication import get_group_size

from ...activations import ACT2FN
from ...cache_utils import (
    Cache,
    DynamicCache,
    get_max_length,
    get_seq_length,
    update,
)
from ...mindspore_utils import ALL_LAYERNORM_LAYERS
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_attn_mask_utils import _MIN_FP16
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import MSPreTrainedModel
from ...generation import GenerationMixin
from ...utils import is_flash_attn_2_available  # Ascend
from ....utils.version_control import check_valid_flash_attention

FLASH_IS_AVAILABLE = is_flash_attn_2_available and check_valid_flash_attention()
if FLASH_IS_AVAILABLE:
    from mindspore.ops.operations.nn_ops import FlashAttentionScore as MSFlashAttention
from ...mindspore_adapter import scaled_dot_product_attention

from mindspore.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Emu3Config"


@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


class Emu3RMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Emu3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = ms.Parameter(ops.ones(hidden_size, dtype=ms.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = (
            hidden_states * ops.rsqrt(variance + self.variance_epsilon) * self.weight
        )
        return hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(Emu3RMSNorm)


class Emu3MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = mint.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = mint.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = mint.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mint.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`ms.Tensor`): The query tensor.
        k (`ms.Tensor`): The key tensor.
        cos (`ms.Tensor`): The cosine part of the rotary embedding.
        sin (`ms.Tensor`): The sine part of the rotary embedding.
        position_ids (`ms.Tensor`):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(ms.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of ops.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to(
        (batch, num_key_value_heads, n_rep, slen, head_dim)
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_construct(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = mint.matmul(query, key_states.swapaxes(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = mint.nn.functional.softmax(
        attn_weights, dim=-1, dtype=ms.float32
    ).to(query.dtype)
    attn_weights = mint.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = mint.matmul(attn_weights, value_states)
    attn_output = attn_output.swapaxes(1, 2).contiguous()  # BN'SD => BSN'D

    return attn_output, attn_weights


ALL_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_construct,
    "sdpa": scaled_dot_product_attention,
}


class Emu3Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Emu3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # self.hidden_size = config.hidden_size  # 4096
        self.num_heads = config.num_attention_heads  # 32
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.num_heads
        )  # 4096 / 32 = 128
        self.num_key_value_heads = config.num_key_value_heads  # 8
        self.num_key_value_groups = (
            self.num_heads // self.num_key_value_heads
        )  # 32 / 8 = 4
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        # self.max_position_embeddings = config.max_position_embeddings  # 9216
        # self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = mint.nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = mint.nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = mint.nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = mint.nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Initialize sequence parallel operator
        # if (sp_group := get_sequence_parallel_group()) is not None:
        #     self.sp_group_size = get_group_size(sp_group)
        #     self.alltoall = ops.AlltoAll(self.sp_group_size, 1, 2, group=sp_group)
        # else:
        self.sp_group_size = None
        self.alltoall = nn.Identity()

        if self.config._attn_implementation == "sdpa": # NOT supported
            self.config._attn_implementation = "eager"
        # Flash Attention
        if self.config._attn_implementation == "flash_attention_2":
            self.enable_flash_attention = FLASH_IS_AVAILABLE
            self.fa_dtype = ms.float16
            if self.enable_flash_attention:  # TODO change to adapter
                dropout_rate = self.attention_dropout if self.training else 0.0
                # sequence parallel
                num_heads = (
                    self.num_heads // self.sp_group_size
                    if self.sp_group_size is not None
                    else self.num_heads
                )
                # Q: (b s n d) -> (b n s d)  #  b - batch_size, s - seq_len, n - num_head, d - head dim
                self.flash_attention = MSFlashAttention(
                    scale_value=self.scaling,
                    head_num=num_heads,
                    keep_prob=1 - dropout_rate,
                    input_layout="BNSD",  # BSH or BNSD
                )
            else:
                self.config._attn_implementation = "eager"

    def construct(
        self,
        hidden_states: ms.Tensor,
        position_embeddings: Tuple[ms.Tensor, ms.Tensor],
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        target_dtype = hidden_states.dtype

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(
            (bsz, q_len, self.num_heads, self.head_dim)
        ).swapaxes(1, 2)
        key_states = key_states.view(
            (bsz, q_len, self.num_key_value_heads, self.head_dim)
        ).swapaxes(1, 2)
        value_states = value_states.view(
            (bsz, q_len, self.num_key_value_heads, self.head_dim)
        ).swapaxes(1, 2)

        # sequence parallel: scatter BNS'D => BN'SD
        query_states = self.alltoall(query_states.float()).to(target_dtype)
        key_states = self.alltoall(key_states.float()).to(target_dtype)
        value_states = self.alltoall(value_states.float()).to(target_dtype)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Attention: BN'SD => BSN'D
        attn_weights = None
        if self.config._attn_implementation == "flash_attention_2":
            attention_interface = self.flash_attention
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        if self.config._attn_implementation == "eager":
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
        elif self.config._attn_implementation == "sdpa":
            attn_output = attention_interface(
                query_states,
                key_states,
                value_states,
                attention_mask,
                dtype=ms.float16
                # dropout=0.0 if not self.training else self.attention_dropout, # NOT yet support
                # scaling=self.scaling,  # NOT yet support
            )
        else:  # flash attention
            # NOTE: MSFlashAttention needs shape of BNSD ==> [batch_size,  num_heads, sequence_length, head_dim].
            if attention_mask is not None:
                attention_mask = self.convert_mask_to_fa_format(attention_mask)
            attn_output = attention_interface(
                query_states.to(self.fa_dtype),
                key_states.to(self.fa_dtype),
                value_states.to(self.fa_dtype),
                None,
                None,
                None,
                attention_mask,
            )[3]
            attn_output = attn_output.to(target_dtype)
            attn_output = attn_output.swapaxes(1, 2)
            # b h n d -> b n h d (bsz, q_len, num_heads, head_dim)

        # sequence parallel: gather BSN'D => BS'ND
        attn_output = self.alltoall(attn_output.float()).to(target_dtype)

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def convert_mask_to_fa_format(self, attention_mask):
        if attention_mask is not None:
            if attention_mask.dtype == ms.bool_:
                # flip mask, since ms FA treats 1 as discard, 0 as retain.
                attention_mask = 1 - attention_mask
                attention_mask = attention_mask.to(ms.uint8)
            else:
                # attention_mask has beed inverted before in _prepare_4d_causal_mask: 0: retain, -inf: discard
                attention_mask = attention_mask.to(ms.float16)
                attention_mask = ops.select(
                    ops.equal(attention_mask, _MIN_FP16),
                    ops.ones((), ms.uint8),
                    ops.zeros((), ms.uint8),
                )

        return attention_mask


class Emu3DecoderLayer(nn.Cell):
    def __init__(self, config: Emu3TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Emu3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Emu3MLP(config)
        self.input_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Emu3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.dropout = mint.nn.Dropout(p=config.attention_dropout)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(ms.Tensor)`, *optional*): cached past key and value projection states
            cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if not output_attentions:
            self_attn_weights = None
        outputs += (self_attn_weights,)

        return outputs


class Emu3VQVAEVectorQuantizer(nn.Cell):
    """
    A module for vector quantization using learned embedding vectors.

    This module implements the quantization process similar to te one described in
    the VQ-VAE (Vector Quantized Variational AutoEncoder) paper. It quantizes continuous
    input vectors into discrete codebook vectors, which are learned during training.
    Current implementation improves over previous ones by avoiding costly matrix multiplications
    and allowing for post-hoc remapping of indices.
    """

    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__()
        self.embedding = mint.nn.Embedding(config.codebook_size, config.embed_dim)
        self.embedding.weight.set_data(
            initializer(
                Uniform(scale=1.0 / config.codebook_size),
                self.embedding.weight.shape,
                self.embedding.weight.dtype,
            )
        )

    def construct(self, hidden_state: ms.Tensor):
        # b t c h w -> b t h w c
        batch_size, temporal, channels, height, width = hidden_state.shape
        hidden_state = hidden_state.permute(
            0, 1, 3, 4, 2
        ).contiguous()  # (b, t, h, w, c)
        hidden_state_flattened = hidden_state.view((-1, channels))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hidden_state_sum = ops.sum(hidden_state_flattened**2, dim=1, keepdim=True)
        embedding_sum = ops.sum(self.embedding.weight**2, dim=1)

        # "bd,dn->bn"
        distances = 2 * ops.matmul(
            hidden_state_flattened, self.embedding.weight.swapaxes(0, 1)
        )
        distances = hidden_state_sum + embedding_sum - distances

        min_encoding_indices = ops.argmin(distances, axis=1)
        min_encoding_indices = min_encoding_indices.view(
            (batch_size, temporal, height, width)
        )
        return min_encoding_indices


class Emu3VQVAEEncoderConvDownsample(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = mint.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def construct(self, hidden_states):
        hidden_states = mint.nn.functional.pad(
            hidden_states, pad=(0, 1, 0, 1), mode="constant", value=0
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAEEncoderConvUpsample(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = mint.nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def construct(self, hidden_states):
        hidden_states = mint.nn.functional.interpolate(
            hidden_states, scale_factor=2.0, mode="nearest"
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAEConv3d(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
        conv3d_dtype=ms.bfloat16,
    ):
        super().__init__()
        padding_sizes = [
            one_kernel - one_stride
            for one_kernel, one_stride in zip(kernel_size[1:], stride[1:])
        ]
        self.padding = ()
        for pad_size in padding_sizes[::-1]:
            self.padding += (pad_size // 2 + pad_size % 2, pad_size // 2)
        self.padding += (2, 0)

        self.conv = mint.nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        ).to_float(conv3d_dtype)

    def construct(self, hidden_states: ms.Tensor):
        origin_dtype = hidden_states.dtype
        hidden_states = mint.nn.functional.pad(hidden_states, self.padding)
        hidden_states = self.conv(hidden_states)
        return hidden_states.to(origin_dtype)


class Emu3VQVAESpatialNorm(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_layer = mint.nn.GroupNorm(
            num_channels=out_channels,
            num_groups=32,
            eps=1e-6,
            affine=True,
        ).to_float(ms.float32)
        self.conv_y = mint.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_b = mint.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def construct(self, hidden_states: ms.Tensor, quant_states: ms.Tensor):
        origin_dtype = hidden_states.dtype
        quant_states = mint.nn.functional.interpolate(
            quant_states, size=hidden_states.shape[-2:], mode="nearest"
        )
        hidden_states = self.norm_layer(hidden_states).to(origin_dtype)
        hidden_states = hidden_states * self.conv_y(quant_states) + self.conv_b(
            quant_states
        )
        return hidden_states


class Emu3VQVAETemporalUpsample(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ):
        super().__init__()
        self.conv = Emu3VQVAEConv3d(
            in_channel,
            out_channel,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
        )

    def construct(self, hidden_states: ms.Tensor):
        batch_size, channels, temporal, height, width = hidden_states.shape
        hidden_states = (
            hidden_states.permute(0, 1, 3, 4, 2)
            .contiguous()
            .view((batch_size, -1, temporal))
        )  # (b, c, h, w, t) => (b, c*h*w, t)
        hidden_states = mint.nn.functional.interpolate(
            hidden_states, scale_factor=2.0, mode="nearest"
        )
        hidden_states = (
            hidden_states.view((batch_size, channels, height, width, -1))
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAETemporalDownsample(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ):
        super().__init__()
        self.conv = Emu3VQVAEConv3d(
            in_channel,
            out_channel,
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
        )

    def construct(self, hidden_states: ms.Tensor):
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAETemporalResnetBlock(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv3d_dtype=ms.bfloat16,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = mint.nn.BatchNorm3d(in_channels).to_float(ms.float32)
        self.conv1 = Emu3VQVAEConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
        )
        self.norm2 = mint.nn.BatchNorm3d(out_channels).to_float(ms.float32)
        self.conv2 = Emu3VQVAEConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = mint.nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            ).to_float(conv3d_dtype)

    def construct(self, hidden_states: ms.Tensor):
        origin_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.norm1(hidden_states).to(origin_dtype)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states).to(origin_dtype)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)
            residual = residual.to(origin_dtype)

        return residual + hidden_states


class Emu3VQVAEResnetBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        quant_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.quant_channels = quant_channels

        if quant_channels is None:
            self.norm1 = mint.nn.GroupNorm(
                num_channels=in_channels, num_groups=32, eps=1e-6, affine=True
            ).to_float(ms.float32)
            self.norm2 = mint.nn.GroupNorm(
                num_channels=out_channels, num_groups=32, eps=1e-6, affine=True
            ).to_float(ms.float32)
        else:
            self.norm1 = Emu3VQVAESpatialNorm(quant_channels, in_channels)
            self.norm2 = Emu3VQVAESpatialNorm(quant_channels, out_channels)

        self.conv1 = mint.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = mint.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = mint.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def construct(
        self, hidden_states: ms.Tensor, quant_channels: Optional[ms.Tensor] = None
    ):
        origin_dtype = hidden_states.dtype
        norm_args = () if self.quant_channels is None else (quant_channels,)

        residual = hidden_states
        hidden_states = self.norm1(hidden_states, *norm_args).to(origin_dtype)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, *norm_args).to(origin_dtype)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)

        return residual + hidden_states


class Emu3VQVAEAttentionBlock(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = mint.nn.Linear(self.embed_dim, self.embed_dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:

        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            (batch_size, q_len, self.num_heads, self.head_dim)
        ).swapaxes(1, 2)
        key_states = key_states.view(
            (batch_size, q_len, self.num_heads, self.head_dim)
        ).swapaxes(1, 2)
        value_states = value_states.view(
            (batch_size, q_len, self.num_heads, self.head_dim)
        ).swapaxes(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = mint.matmul(query_states, key_states.swapaxes(2, 3)) * self.scale

        if attn_weights.shape != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(
            query_states.dtype
        )
        attn_weights = mint.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = mint.matmul(attn_weights, value_states)

        if attn_output.shape != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Emu3VQVAEGroupNorm(mint.nn.GroupNorm):
    """
    Same as the GroupNorm with the only difference that this ones accepts
    an optional kwarg `quant_states` which is not used. This class makes it easier to
    use SpatialNorm or GroupNorm without conditionals
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def construct(self, input, quant_states=None):
        origin_dtype = input.dtype
        return mint.nn.functional.group_norm(
            input.float(), self.num_groups, self.weight, self.bias, self.eps
        ).to(origin_dtype)


class Emu3VQVAEMiddleBlock(nn.Cell):
    def __init__(self, config, in_channels, quant_channels=None):
        super().__init__()

        self.block_1 = Emu3VQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            quant_channels=quant_channels,
        )
        self.attn_1 = Emu3VQVAEAttentionBlock(config)
        if quant_channels is None:
            self.attn_norm = Emu3VQVAEGroupNorm(
                num_channels=in_channels, num_groups=32, eps=1e-6, affine=True
            )
        else:
            self.attn_norm = Emu3VQVAESpatialNorm(quant_channels, in_channels)

        self.block_2 = Emu3VQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            quant_channels=quant_channels,
        )

    def construct(self, hidden_states: ms.Tensor, quant_states: ms.Tensor = None):
        hidden_states = self.block_1(hidden_states, quant_states)
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states, quant_states)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(
            (batch_size, channels, height * width)
        ).swapaxes(1, 2)
        hidden_states = self.attn_1(hidden_states)[0]
        hidden_states = hidden_states.reshape(
            batch_size, height, width, channels
        ).permute((0, 3, 1, 2))
        hidden_states = residual + hidden_states
        hidden_states = self.block_2(hidden_states, quant_states)
        return hidden_states


class Emu3VQVAEDownBlock(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        channel_multiplier = config.channel_multiplier

        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.CellList()
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            attn_norms = nn.CellList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Emu3VQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if (
                    config.attn_resolutions is not None
                    and i_level in config.attn_resolutions
                ):
                    attn.append(Emu3VQVAEAttentionBlock(config))
                    attn_norms.append(
                        mint.nn.GroupNorm(
                            num_channels=block_in, num_groups=32, eps=1e-6, affine=True
                        ).to_float(ms.float32)
                    )

            down = nn.Cell()
            down.block = block
            down.attn = attn
            down.attn_norms = attn_norms
            if i_level != self.num_resolutions - 1:
                down.downsample = Emu3VQVAEEncoderConvDownsample(block_in)
            self.down.append(down)

    def construct(self, hidden_states: ms.Tensor):
        origin_dtype = hidden_states.dtype
        for i_level, blocks in enumerate(self.down):
            for i_block in range(self.num_res_blocks):
                hidden_states = blocks.block[i_block](hidden_states)
                if len(blocks.attn) > 0:
                    residual = hidden_states
                    hidden_states = blocks.attn_norms[i_block](hidden_states).to(origin_dtype)

                    batch_size, channels, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(
                        (batch_size, channels, height * width)
                    ).swapaxes(1, 2)
                    hidden_states = blocks.attn[i_block](hidden_states)[0]

                    hidden_states = hidden_states.reshape(
                        batch_size, height, width, channels
                    ).permute(0, 3, 1, 2)
                    hidden_states = residual + hidden_states

            if i_level != self.num_resolutions - 1:
                hidden_states = blocks.downsample(hidden_states)

        return hidden_states


class Emu3VQVAEUpBlock(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks

        quant_channels = config.embed_dim
        block_in = config.base_channels * config.channel_multiplier[-1]

        self.up = nn.CellList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            attn_norms = nn.CellList()
            block_out = config.base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Emu3VQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        quant_channels=quant_channels,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VQVAEAttentionBlock(config))
                    attn_norms.append(Emu3VQVAESpatialNorm(quant_channels, block_in))

            up = nn.Cell()
            up.block = block
            up.attn = attn
            up.attn_norms = attn_norms
            if i_level != 0:
                up.upsample = Emu3VQVAEEncoderConvUpsample(block_in)

            self.up.insert(0, up)

    def construct(self, hidden_states: ms.Tensor, quant_states: ms.Tensor):
        for i_level, blocks in enumerate(self.up[::-1]):
            for i_block in range(self.num_res_blocks + 1):
                hidden_states = blocks.block[i_block](hidden_states, quant_states)
                if len(blocks.attn) > 0:
                    residual = hidden_states
                    hidden_states = blocks.attn_norms[i_block](
                        hidden_states, quant_states
                    )

                    batch_size, channels, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(
                        (batch_size, channels, height * width)
                    ).swapaxes(1, 2)
                    hidden_states = blocks.attn[i_block](hidden_states)[0]

                    hidden_states = hidden_states.reshape(
                        batch_size, height, width, channels
                    ).permute(0, 3, 1, 2)
                    hidden_states = residual + hidden_states
            if i_level != len(self.up) - 1:
                hidden_states = blocks.upsample(hidden_states)

        return hidden_states


class Emu3VQVAEEncoder(nn.Cell):
    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__()
        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier
        out_channels = 2 * latent_channels if double_latent else latent_channels
        block_in = base_channels * channel_multiplier[-1]

        self.conv_in = mint.nn.Conv2d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1
        )
        self.down_block = Emu3VQVAEDownBlock(config)  # downsample
        self.middle_block = Emu3VQVAEMiddleBlock(config, block_in)  # middle

        self.norm_out = mint.nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        ).to_float(ms.float32)  # end

        self.conv_out = mint.nn.Conv2d(
            block_in,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        temporal_down_blocks = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.CellList()
        self.time_res_stack = nn.CellList()

        for i in range(temporal_down_blocks):
            conv = Emu3VQVAETemporalDownsample(out_channels, out_channels)
            self.time_conv.append(conv)

        for _ in range(config.num_res_blocks):
            time_res_conv = Emu3VQVAETemporalResnetBlock(
                in_channels=out_channels,
                out_channels=out_channels,
            )
            self.time_res_stack.append(time_res_conv)

    def construct(self, pixel_values: ms.Tensor):
        origin_dtype = pixel_values.dtype

        temporal_dim = pixel_values.shape[1]
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])

        # downsampling & middle
        hidden_states = self.conv_in(pixel_values)
        hidden_states = self.down_block(hidden_states)
        hidden_states = self.middle_block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states).to(origin_dtype)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        hidden_states = hidden_states.reshape(
            -1, temporal_dim, *hidden_states.shape[1:]
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        # temporal convs
        for conv in self.time_conv:
            hidden_states = conv(hidden_states)
            hidden_states *= mint.sigmoid(hidden_states)

        for layer in self.time_res_stack:
            hidden_states = layer(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        return hidden_states


class Emu3VQVAEDecoder(nn.Cell):
    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__()

        quant_channels = config.embed_dim
        block_in = config.base_channels * config.channel_multiplier[-1]
        self.time_res_stack = nn.CellList()
        for _ in range(config.num_res_blocks):
            time_res_conv = Emu3VQVAETemporalResnetBlock(
                in_channels=config.latent_channels, out_channels=config.latent_channels
            )
            self.time_res_stack.append(time_res_conv)

        temp_upsample_block_num = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.CellList()
        for i in range(temp_upsample_block_num):
            conv = Emu3VQVAETemporalUpsample(
                config.latent_channels, config.latent_channels
            )
            self.time_conv.append(conv)

        self.conv_in = mint.nn.Conv2d(
            config.latent_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.middle_block = Emu3VQVAEMiddleBlock(
            config, block_in, quant_channels=quant_channels
        )
        self.up_block = Emu3VQVAEUpBlock(config)

        block_in = config.base_channels * config.channel_multiplier[0]
        self.norm_out = Emu3VQVAESpatialNorm(quant_channels, block_in)

        self.conv_out = mint.nn.Conv2d(
            block_in,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def construct(self, hidden_states: ms.Tensor, quant_states: ms.Tensor):
        hidden_quant_states = mint.cat((hidden_states, quant_states), dim=0)
        hidden_quant_states = hidden_quant_states.permute(0, 2, 1, 3, 4)
        # temporal convs
        for layer in self.time_res_stack:
            hidden_quant_states = layer(hidden_quant_states)

        for layer in self.time_conv:
            hidden_quant_states = layer(hidden_quant_states)
            hidden_quant_states *= mint.sigmoid(hidden_quant_states)

        hidden_quant_states = hidden_quant_states.permute(0, 2, 1, 3, 4)
        hidden_states, quant_states = mint.chunk(hidden_quant_states, 2, dim=0)
        hidden_states = hidden_states.reshape(-1, *hidden_states.shape[2:])
        quant_states = quant_states.reshape(-1, *quant_states.shape[2:])

        hidden_states = self.conv_in(hidden_states)

        # middle & upsampling
        hidden_states = self.middle_block(hidden_states, quant_states)
        hidden_states = self.up_block(hidden_states, quant_states)

        hidden_states = self.norm_out(hidden_states, quant_states)
        hidden_states *= mint.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


EMU3_VQ_START_DOCSTRING = r"""
    This model inherits from [`MSPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a MindSpore [nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3VQVAEConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~MSPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    """The VQ-VAE model used in Emu3 for encoding/decoding images into discrete tokens.
    This model follows the "Make-a-scene: Scene-based text-to-image generation with human priors" paper from
    [ Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman](https://arxiv.org/abs/2203.13131).
    """,
    EMU3_VQ_START_DOCSTRING,
)
def _calculate_fan_in_and_fan_out(arr):
    # calculate fan_in and fan_out. fan_in is the number of input units in `arr` , and fan_out is the number of output units in `arr`.
    shape = arr.shape
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError(
            "'fan_in' and 'fan_out' can not be computed for arr with fewer than"
            " 2 dimensions, but got dimensions {}.".format(dimensions)
        )
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        for i in range(2, dimensions):
            receptive_field_size *= shape[i]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


class Emu3VQVAE(MSPreTrainedModel):
    config_class = Emu3VQVAEConfig
    base_model_prefix = "emuvideovq"
    main_input_name = "pixel_values"
    _no_split_modules = [
        "Emu3VQVAETemporalResnetBlock",
        "Emu3VQVAEAttentionBlock",
        "Emu3VQVAEResnetBlock",
        "Emu3VQVAEVectorQuantizer",
    ]

    def _init_weights(self, module):
        if isinstance(module, (mint.nn.Conv2d, mint.nn.Conv3d)):
            module.weight.set_data(
                initializer(
                    HeNormal(mode="fan_out", nonlinearity="relu"),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
        elif isinstance(module, mint.nn.Linear):
            weight = initializer(
                HeNormal(negative_slope=math.sqrt(5)),
                module.weight.shape,
                module.weight.dtype,
            )
            module.weight.set_data(weight)
            if module.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                bias_weight = initializer(
                    Uniform(scale=bound), module.bias.shape, module.bias.dtype
                )
                module.bias.set_data(bias_weight)
        elif isinstance(module, (mint.nn.BatchNorm2d, mint.nn.GroupNorm)):
            module.weight.set_data(
                initializer(Constant(1), module.weight.shape, module.weight.dtype)
            )
            module.bias.set_data(
                initializer(Constant(0), module.bias.shape, module.bias.dtype)
            )
        elif isinstance(module, mint.nn.BatchNorm3d):
            module.weight.set_data(
                initializer(Constant(1), module.weight.shape, module.weight.dtype)
            )
            module.bias.set_data(
                initializer(Constant(0), module.bias.shape, module.bias.dtype)
            )

    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__(config)

        self.config = config

        self.encoder = Emu3VQVAEEncoder(config)
        self.decoder = Emu3VQVAEDecoder(config)
        self.quantize = Emu3VQVAEVectorQuantizer(config)
        self.vision_spatial_factor = 2 ** (len(config.channel_multiplier) - 1)

        self.quant_conv = Emu3VQVAEConv3d(
            config.latent_channels,
            config.embed_dim,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
        )
        self.post_quant_conv = Emu3VQVAEConv3d(
            config.embed_dim,
            config.latent_channels,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
        )
        self.spatial_scale_factor = 2 ** (len(config.channel_multiplier) - 1)
        self.set_train(False)  # Emu3's VQ model is frozen

        self.post_init()

    def encode(self, pixel_values: ms.Tensor, image_sizes: ms.Tensor):
        is_image = pixel_values.ndim == 4
        if is_image:
            temporal = self.config.temporal_downsample_factor
            batch_size, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.unsqueeze(1).tile((1, temporal, 1, 1, 1))
        else:
            batch_size, temporal, channels, height, width = pixel_values.shape

        hidden_states = self.encoder(pixel_values)

        # b t c h w -> b c t h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = self.quant_conv(hidden_states)

        # b c t h w -> b t c h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        codes = self.quantize(hidden_states)

        image_tokens = codes.squeeze(1) if is_image else codes

        image_tokens = [
            single_image[
                : int(size[0] / self.vision_spatial_factor),
                : int(size[1] / self.vision_spatial_factor),
            ]
            for single_image, size in zip(image_tokens, image_sizes)
        ]

        return image_tokens

    def decode(self, hidden_states: ms.Tensor):
        is_image = hidden_states.ndim == 3
        if is_image:
            hidden_states = hidden_states.unsqueeze(1)

        batch_size, temporal, height, width = hidden_states.shape
        quant = self.quantize.embedding(hidden_states.flatten(start_dim=0))

        channels = quant.shape[-1]
        quant = (
            quant.view((batch_size, temporal, height, width, channels))
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        post_quant = self.post_quant_conv(quant)

        quant = quant.permute(0, 2, 1, 3, 4)
        post_quant = post_quant.permute(0, 2, 1, 3, 4)

        video = self.decoder(post_quant, quant)
        video = video.reshape(
            batch_size,
            temporal * self.config.temporal_downsample_factor,
            self.config.out_channels,
            height * self.spatial_scale_factor,
            width * self.spatial_scale_factor,
        )
        return video[:, 0] if is_image else video


class Emu3ImageVocabularyMapping:
    """
    A class for mapping discrete image tokens from VQGAN to BPE tokens.
    """

    def __init__(self, vocab_map):
        self.vocab_map = vocab_map
        self.eol_token_id = vocab_map.get("<|extra_200|>")
        self.image_token_id = vocab_map.get("<image>")

    @cached_property
    def image_tokens(self):
        return sorted(
            [
                val
                for name, val in self.vocab_map.items()
                if name.startswith("<|visual token")
            ]
        )

    @cached_property
    def image_tokens_str(self):
        return sorted(
            [
                name
                for name, val in self.vocab_map.items()
                if name.startswith("<|visual token")
            ]
        )

    @cached_property
    def img2bpe(self):
        return {
            int(token[-8:-2]): self.vocab_map[token] for token in self.image_tokens_str
        }

    @cached_property
    def bpe2img(self):
        return {v: k for k, v in self.img2bpe.items()}

    @cached_property
    def bpe2img_mapping_tensor(self):
        mapping = mint.zeros(max(self.bpe2img.keys()) + 1, dtype=ms.int32)
        for k, v in self.bpe2img.items():
            mapping[k] = v
        return mapping

    @cached_property
    def img2bpe_mapping_tensor(self):
        mapping = mint.zeros(max(self.img2bpe.keys()) + 1, dtype=ms.int32)
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

    def convert_img2bpe(self, img_batch: List[ms.Tensor]) -> ms.Tensor:
        eol_row = mint.ones((img_batch.shape[0], 1), dtype=ms.int32) * self.eol_token_id
        img_tokens = self.img2bpe_mapping_tensor[img_batch]
        img_tokens = mint.cat([img_tokens, eol_row], dim=-1)
        return img_tokens

    def convert_bpe2img(self, img_batch: ms.Tensor) -> ms.Tensor:
        img_batch = img_batch[..., :-1]  # remove last row of EOL tokens
        img_tokens = self.bpe2img_mapping_tensor[img_batch]
        return img_tokens


EMU3_START_DOCSTRING = r"""
    This model inherits from [`MSPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a mindspore.nn.Cell subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~MSPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3PreTrainedModel(MSPreTrainedModel):
    config_class = Emu3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Emu3DecoderLayer",
    ]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_quantized_cache = True
    _supports_cache_class = (
        True  # support Cache Classes; True: use DynamicCache by default
    )
    _supports_static_cache = False
    _supports_param_buffer_assignment = False
    _supports_flex_attn = False

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        if isinstance(module, mint.nn.Linear):
            module.weight.set_data(
                initializer(
                    Normal(sigma=std, mean=0.0),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
            if module.bias is not None:
                module.bias.set_data(
                    initializer("zeros", module.bias.shape, module.bias.dtype)
                )
        elif isinstance(module, mint.nn.Embedding):
            module.weight.set_data(
                initializer(
                    Normal(sigma=std, mean=0.0),
                    module.weight.shape,
                    module.weight.dtype,
                )
            )
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0


class Emu3RotaryEmbedding(nn.Cell):
    def __init__(self, config: Emu3Config):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"  # currenly only support "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = mint.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, seq_len=seq_len
            )
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len

    def construct(self, x, position_ids):
        with no_grad():
            if "dynamic" in self.rope_type:  # not used
                self._dynamic_frequency_update(position_ids)

            # Core RoPE block
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .broadcast_to((position_ids.shape[0], -1, 1))
            )
            position_ids_expanded = position_ids[:, None, :].float()
            # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).swapaxes(1, 2)
            emb = mint.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
            cos = cos * self.attention_scaling
            sin = sin * self.attention_scaling

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


EMU3_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(ms.Tensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(ms.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3TextModel(Emu3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Emu3TextDecoderLayer`]

    Args:
        config: Emu3TextConfig
    """

    def __init__(self, config: Emu3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = mint.nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.CellList(
            [
                Emu3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Emu3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize sequence parallel
        # if (sp_group := get_sequence_parallel_group()) is not None:
        #     logger.info(f"Initialize Emu3 model with sequence parallel group `{sp_group}`.")
        #     self.split_forward_gather_backward = SplitFowardGatherBackward(dim=1, grad_scale="down", group=sp_group)
        #     self.gather_forward_split_backward = GatherFowardSplitBackward(dim=1, grad_scale="up", group=sp_group)
        # else:
        self.split_forward_gather_backward = nn.Identity()
        self.gather_forward_split_backward = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def recompute(self, cell, **recompute_kwargs):
        if isinstance(cell, mint.nn.Dropout):
            return
        if not cell._has_config_recompute:
            cell.recompute(**recompute_kwargs)
        if isinstance(cell, nn.CellList):
            self.recompute(cell[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            cell.add_flags(output_no_recompute=True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        if gradient_checkpointing_kwargs is None:
            # gradient_checkpointing_kwargs = {"mp_comm_recompute": True, "parallel_optimizer_comm_recompute": True}
            gradient_checkpointing_kwargs = {}

        # llama layers
        for decoder_layer in self.layers:
            assert isinstance(decoder_layer, Emu3DecoderLayer)
            for name, cell in decoder_layer.name_cells().items():
                self.recompute(cell, **gradient_checkpointing_kwargs)
        self.recompute(self.embed_tokens, **gradient_checkpointing_kwargs)
        self.recompute(self.norm, **gradient_checkpointing_kwargs)

        logger.info(f"{self.__class__.__name__}: enable recompute.")

    @add_start_docstrings_to_model_forward(EMU3_TEXT_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # sequence parallel start: BxSxD => BxS'xD (S'=S/M)
        inputs_embeds = self.split_forward_gather_backward(inputs_embeds)  # BSD => BS'D

        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    past_seen_tokens = past_key_values.get_seq_length()
                else:  # tuple static cache
                    past_seen_tokens = get_seq_length(past_key_values)
            cache_position = ops.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                dtype=ms.int32,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # sequence parallel end
        hidden_states = self.gather_forward_split_backward(hidden_states)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: ms.Tensor,
        input_tensor: ms.Tensor,
        cache_position: ms.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        past_seen_tokens = 0
        if past_key_values is not None:
            past_seen_tokens = (
                get_seq_length(past_key_values)
                if isinstance(past_key_values, tuple)
                else past_key_values.get_seq_length()
            )
        using_static_cache = isinstance(past_key_values, tuple)

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = get_max_length(past_key_values)
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, ms.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: ms.Tensor,
        sequence_length: int,
        target_length: int,
        cache_position: ms.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`ms.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            cache_position (`ms.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`ms.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.ndim == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = _MIN_FP16
            causal_mask = ops.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=ms.float16
            )
            if sequence_length != 1:
                causal_mask = mint.triu(causal_mask, diagonal=1)
            causal_mask *= ops.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].broadcast_to(
                (batch_size, 1, -1, -1)
            )
            if attention_mask is not None:
                causal_mask = causal_mask.clone() # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class Emu3ForCausalLM(Emu3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Emu3TextConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Emu3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = mint.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_func = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class="Emu3TextConfig"
    )
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
        logits_to_keep: Union[int, ms.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import Emu3Processor
        >>> from mindway.transformers import Emu3ForCausalLM
        >>> from mindspore import Tensor

        >>> model = Emu3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_EMU3_WEIGHTS)
        >>> processor = Emu3Processor.from_pretrained(PATH_TO_CONVERTED_EMU3_WEIGHTS)

        >>> inputs = processor(text=["Can you write me a poem about winter."], return_tensors="np")
        >>> generated_ids = model.generate(Tensor(inputs.input_ids), max_new_tokens=100, do_sample=False)
        >>> answer = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits.float()  # some logits processor not support bf16

        loss = None
        if labels is not None:  # training pipeline
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view((-1, self.config.vocab_size))
            shift_labels = shift_labels.view((-1))
            loss = self.loss_func(shift_logits, shift_labels)
            return loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


EMU3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`ms.Tensor` of shape `(batch_size, max_num_images, max_num_tiles, channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Emu3ImageProcessor.__call__`] for details ([]`Emu3Processor`] uses
            [`Emu3ImageProcessor`] for processing images).
        image_sizes (`ms.Tensor` of shape `(batch_size, 2)`):
                The sizes of the images in the batch, being (height, width) for each image. Image sizes can be obtained using
            [`AutoImageProcessor`]. See [`Emu3ImageProcessor.__call__`] for details ([]`Emu3Processor`] uses
            [`Emu3ImageProcessor`] for processing images).
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(ms.Tensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Has to be an instance of [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class Emu3ForConditionalGeneration(Emu3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["text_model.lm_head.weight"]
    _supports_static_cache = False  # `get_image_tokens()`, called when `pixel_values` is passed, is not compileable

    def __init__(self, config):
        super().__init__(config)
        self.text_model = Emu3ForCausalLM(config.text_config)
        self.vqmodel = Emu3VQVAE(config.vq_config)
        self.vocabulary_mapping = Emu3ImageVocabularyMapping(config.vocabulary_map)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_image_tokens(self, pixel_values: ms.Tensor, image_sizes: ms.Tensor):
        """
        Tokenizes images into discrete tokens with VQGAN module. Converts
        obtained image tokens into BPE tokens and wraps with "boi" and "eoi"
        special tokens.

        Args:
            pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_sizes (`ms.Tensor` of shape `(batch_size, 2)`):
                The sizes of the images in the batch, being (height, width) for each image.
        """
        image_tokens_list = self.vqmodel.encode(pixel_values, image_sizes)
        bpe_tokens_list = [
            self.vocabulary_mapping.convert_img2bpe(tokens).flatten(start_dim=0)
            for tokens in image_tokens_list
        ]
        bpe_tokens = mint.cat(bpe_tokens_list)
        return bpe_tokens

    def decode_image_tokens(self, image_tokens: ms.Tensor, height: int, width: int):
        """
        Decodes generated image tokens from language model to continuous pixel values
        with VQGAN module via upsampling.

        Args:
            image_tokens (`ms.Tensor` of shape `(batch_size, num_of_tokens)`):
                The tensors corresponding to the input images.
            height (`int`):
                Height of the generated image before upsampling.
            width (`int`):
                Width of the generated image before upsampling.
        """
        with no_grad():
            sequences = image_tokens[:, :-3].view((-1, height, width + 1))
            image_tokens = ops.stop_gradient(self.vocabulary_mapping.convert_bpe2img(sequences))
            image = ops.stop_gradient(self.vqmodel.decode(image_tokens))
            return image

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def construct(
        self,
        input_ids: ms.Tensor = None,
        pixel_values: ms.Tensor = None,
        image_sizes: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        logits_to_keep: Union[int, ms.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `ms.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `ms.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import Emu3Processor
        >>> from mindway.transformers import Emu3ForConditionalGeneration
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> import requests
        >>> from PIL import Image

        >>> model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", mindspore_dtype=ms.bfloat16)
        >>> processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")

        >>> conversation = [
        ...     {
        ...     "role": "system",
        ...     "content": [
        ...         {"type": "text", "text": "You are a helpful assistant."},
        ...         ],
        ...     },
        ...     {
        ...     "role": "user",
        ...     "content": [
        ...         {"type": "image"},
        ...         {"type": "text", "text": "Please describe the image."},
        ...         ],
        ...     },
        ... ]

        >>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        >>> image = Image.open(requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw)

        >>> inputs = processor(images=[image], text=[prompt], return_tensors="np").to(ms.bfloat16)

        >>> generated_ids = model.generate(Tensor(inputs.input_ids), max_new_tokens=100, do_sample=False)
        >>> answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is not None) and (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None:
            image_tokens = self.get_image_tokens(pixel_values, image_sizes)
            special_image_mask = input_ids == self.vocabulary_mapping.image_token_id
            image_tokens = image_tokens.to(input_ids.dtype)
            input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        **kwargs
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            use_cache=use_cache,
            **kwargs
        )

        if (cache_position is not None) and cache_position[0] != 0:
            model_inputs["pixel_values"] = None

        return model_inputs


__all__ = [
    "Emu3ForConditionalGeneration",
    "Emu3ForCausalLM",
    "Emu3TextModel",
    "Emu3PreTrainedModel",
    "Emu3VQVAE",
]
