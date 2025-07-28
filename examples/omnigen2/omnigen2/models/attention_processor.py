"""
OmniGen2 Attention Processor Module

Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
from typing import Optional

from einops import repeat

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Tensor, mint, ops

from mindone.diffusers.models.attention_processor import Attention

from .embeddings import apply_rotary_emb
from .fa_utils import index_first_axis, pad_input, unpad_input


class OmniGen2AttnProcessorFlash2Varlen:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling
    """

    def _upad_input(
        self,
        query_layer: Tensor,
        key_layer: Tensor,
        value_layer: Tensor,
        attention_mask: Tensor,
        query_length: int,
        num_heads: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[Tensor, Tensor], tuple[int, int]]:
        """
        Unpad the input tensors for flash attention.

        Args:
            query_layer: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            key_layer: Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            value_layer: Value tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            attention_mask: Attention mask tensor of shape (batch_size, seq_len)
            query_length: Length of the query sequence
            num_heads: Number of attention heads

        Returns:
            Tuple containing:
                - Unpadded query tensor
                - Unpadded key tensor
                - Unpadded value tensor
                - Query indices
                - Tuple of cumulative sequence lengths for query and key
                - Tuple of maximum sequence lengths for query and key
        """

        def _get_unpad_data(attention_mask: Tensor) -> tuple[Tensor, Tensor, int]:
            """Helper function to get unpadding data from attention mask."""
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=ms.int32)
            indices = mint.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(mint.cumsum(seqlens_in_batch, dim=0, dtype=ms.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # Unpad key and value layers
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # Handle different query length cases
        if query_length == kv_seq_len:
            query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = mint.arange(batch_size + 1, dtype=ms.int32)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> Tensor:
        """
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            Tensor: Processed hidden states after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # Unpad input for flash attention
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query, key, value, attention_mask, sequence_length, attn.heads
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens

        # Handle different number of heads
        if kv_heads < attn.heads:
            key_states = repeat(key_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)
            value_states = repeat(value_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)

        # Apply flash attention
        attn_output_unpad = ops.flash_attention_score(
            query_states,
            key_states,
            value_states,
            head_num=attn.heads,
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_k,
            scalar_value=softmax_scale,
            input_layout="TND",
        )

        # Pad output and apply final transformations
        hidden_states = pad_input(attn_output_unpad, indices_q, batch_size, sequence_length)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class OmniGen2AttnProcessor:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> Tensor:
        """
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            Tensor: Processed hidden states after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # explicitly repeat key and value to match query length
        key = key.repeat_interleave(query.shape[-3] // key.shape[-3], -3)
        value = value.repeat_interleave(query.shape[-3] // value.shape[-3], -3)

        hidden_states = ops.speed_fusion_attention(
            query, key, value, head_num=attn.heads, input_layout="BNSD", atten_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
