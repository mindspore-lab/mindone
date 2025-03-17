from typing import Optional, Tuple

import mindspore as ms
from mindspore import mint, ops
from mindspore.communication import get_group_size

from mindone.diffusers.models.attention_processor import Attention
from mindone.diffusers.utils import logging

from ..acceleration import AlltoAll, get_sequence_parallel_group

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@ms.jit_class
class CogVideoXAttnProcessor2_0_SP:
    r"""
    Sequence Parallel Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, heads, scale, enable_sequence_parallelism=False):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from mindone.diffusers.models.embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if enable_sequence_parallelism:
            self.sp_group = get_sequence_parallel_group()
            self.sp_size = get_group_size(self.sp_group)
            self.fa = ops.operations.nn_ops.FlashAttentionScore(
                head_num=heads // self.sp_size,
                scale_value=scale,
                input_layout="BNSD",
            )
            self.alltoall = AlltoAll(split_dim=1, concat_dim=2, group=self.sp_group)
            self.alltoall_back = AlltoAll(split_dim=2, concat_dim=1, group=self.sp_group)
        else:
            self.fa = ops.operations.nn_ops.FlashAttentionScore(head_num=heads, scale_value=scale, input_layout="BNSD")

    def apply_rotary_emb_for_image_part(
        self,
        hidden_state: ms.Tensor,
        image_rotary_emb: ms.Tensor,
        start_index: int,
        axis: int = 2,
    ) -> ms.Tensor:
        """
        Equivalence of expression(when axis=2):
            `hidden_state[:, :, start_index:] = self.apply_rotary_emb(hidden_state[:, :, start_index:], image_rotary_emb)`

        Rewrite it since implement above might call ops.ScatterNdUpdate which is super slow!
        """
        hidden_state_text, hidden_state_image = mint.split(
            hidden_state, (start_index, hidden_state.shape[axis] - start_index), dim=axis
        )
        hidden_state_image = self.apply_rotary_emb(hidden_state_image, image_rotary_emb)
        hidden_state = mint.cat([hidden_state_text, hidden_state_image], dim=axis)
        return hidden_state

    def _rearange_input(self, hidden_states, to_func, n):
        x = to_func(hidden_states)
        b, s = x.shape[0], x.shape[1]
        # It is not feasible to replace `x.view(b, s, n, x.shape[0] // n)` with `x.view(b, s, n, -1)`. Otherwise, a
        # "RuntimeError" will be thrown in the dynamic shape scenario: "For Chunk, the dimension corresponds to the
        # specified 'dims' is dynamic, which is not supported now".
        x = x.view(b, s, n, x.shape[-1] // n).swapaxes(1, 2)
        if self.enable_sequence_parallelism:
            # b, n, sub_s, d -> b, sub_n, s, d
            x = self.alltoall(x)
        return x

    def _rearange_output(self, x, attn):
        if self.enable_sequence_parallelism:
            # b, sub_n, s, d -> b, n, sub_s, d
            x = self.alltoall_back(x)
        b, n, s, d = x.shape
        x = x.swapaxes(1, 2).reshape(b, -1, n * d)
        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        return x

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = self._rearange_input(hidden_states, attn.to_q, attn.heads)
        key = self._rearange_input(hidden_states, attn.to_k, attn.heads)
        value = self._rearange_input(hidden_states, attn.to_v, attn.heads)

        encoder_query = self._rearange_input(encoder_hidden_states, attn.to_q, attn.heads)
        encoder_key = self._rearange_input(encoder_hidden_states, attn.to_k, attn.heads)
        encoder_value = self._rearange_input(encoder_hidden_states, attn.to_v, attn.heads)

        text_seq_length = encoder_query.shape[2]

        query = mint.cat([encoder_query, query], dim=2)
        key = mint.cat([encoder_key, key], dim=2)
        value = mint.cat([encoder_value, value], dim=2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        # rewrite the implement for performance, refer to `self.apply_rotary_emb_for_image_part`
        if image_rotary_emb is not None:
            query = self.apply_rotary_emb_for_image_part(query, image_rotary_emb, text_seq_length)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb_for_image_part(key, image_rotary_emb, text_seq_length)
        if not (attn.fa_op_available and attn._enable_flash_sdp):
            hidden_states = attn.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            hidden_states = self.fa(query, key, value, None, None, None, attention_mask)[3]

        encoder_hidden_states = hidden_states[:, :, :text_seq_length, :]
        hidden_states = hidden_states[:, :, text_seq_length:, :]
        hidden_states = self._rearange_output(hidden_states, attn)
        encoder_hidden_states = self._rearange_output(encoder_hidden_states, attn)

        return hidden_states, encoder_hidden_states


@ms.jit_class
class FusedCogVideoXAttnProcessor2_0_SP:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, heads, scale, enable_sequence_parallelism=False):
        # move importing from __call__ to __init__ as it is not supported in construct()
        from mindone.diffusers.models.embeddings import apply_rotary_emb

        self.apply_rotary_emb = apply_rotary_emb
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if enable_sequence_parallelism:
            self.sp_group = get_sequence_parallel_group()
            self.sp_size = get_group_size(self.sp_group)
            self.fa = ops.operations.nn_ops.FlashAttentionScore(
                head_num=heads // self.sp_size, scale_value=scale, input_layout="BNSD"
            )
            self.alltoall = AlltoAll(split_dim=1, concat_dim=2, group=self.sp_group)
            self.alltoall_back = AlltoAll(split_dim=2, concat_dim=1, group=self.sp_group)
        else:
            self.fa = ops.operations.nn_ops.FlashAttentionScore(head_num=heads, scale_value=scale, input_layout="BNSD")

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        image_rotary_emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = mint.split(qkv, split_size, dim=-1)

        encoder_qkv = attn.to_qkv(encoder_hidden_states)
        encoder_split_size = encoder_qkv.shape[-1] // 3
        encoder_query, encoder_key, encoder_value = mint.split(encoder_qkv, encoder_split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if self.enable_sequence_parallelism:
            # b, h, sub_n, d -> b, sub_h, n, d
            query, key, value = self.alltoall(query), self.alltoall(key), self.alltoall(value)
            encoder_query, encoder_key, encoder_value = (
                self.alltoall(encoder_query),
                self.alltoall(encoder_key),
                self.alltoall(encoder_value),
            )

        text_seq_length = encoder_query.shape[2]

        query = mint.cat([encoder_query, query], dim=2)
        key = mint.cat([encoder_key, key], dim=2)
        value = mint.cat([encoder_value, value], dim=2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        # Apply RoPE if needed
        if image_rotary_emb is not None:
            q = query[:, :, text_seq_length:]
            q = self.apply_rotary_emb(q[:, :, text_seq_length:], image_rotary_emb)
            query = mint.cat((query[:, :, :text_seq_length], q), dim=2)
            if not attn.is_cross_attention:
                k = key[:, :, text_seq_length:]
                k = self.apply_rotary_emb(k, image_rotary_emb)
                key = mint.cat((key[:, :, :text_seq_length], k), dim=2)
        if not (attn.fa_op_available and attn._enable_flash_sdp):
            hidden_states = attn.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = self.fa(query, key, value, None, None, None, attention_mask)[3]

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[2] - text_seq_length], axis=2
        )

        if self.enable_sequence_parallelism:
            # b, sub_h, n, d -> b, h, sub_n, d
            encoder_hidden_states = self.alltoall_back(encoder_hidden_states)
            hidden_states = self.alltoall_back(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # linear proj
        encoder_hidden_states = attn.to_out[0](encoder_hidden_states)
        # dropout
        encoder_hidden_states = attn.to_out[1](encoder_hidden_states)
        return hidden_states, encoder_hidden_states
