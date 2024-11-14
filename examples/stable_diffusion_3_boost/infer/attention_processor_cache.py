from typing import Optional
import math
import mindspore as ms
from mindspore import ops

from mindone.diffusers.models.attention_processor import Attention
from mindone.diffusers.models.attention_processor import JointAttnProcessor
from mindone.diffusers.models.attention import _chunked_feed_forward


def joint_transformerblock_construct(self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor, temb: ms.Tensor):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (None,) * 5
    if self.context_pre_only:
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
    else:
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
    # Attention.
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
    )
    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp.expand_dims(1)) + shift_mlp.expand_dims(1)
    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
    else:
        ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output
    hidden_states = hidden_states + ff_output
    # Process attention outputs for the `encoder_hidden_states`.
    if self.context_pre_only:
        encoder_hidden_states = None
    else:
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp.expand_dims(1)) + c_shift_mlp.expand_dims(1)
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            context_ff_output = _chunked_feed_forward(
                self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    return encoder_hidden_states, hidden_states


def downsample(hidden_states, merge_factor, method='nearest'):
    batch_size, _, channel = hidden_states.shape
    cur_h = int(math.sqrt(hidden_states.shape[1]))
    cur_w = cur_h
    new_h, new_w = int(cur_h / merge_factor), int(cur_w / merge_factor)
    hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, channel, cur_h, cur_w)
    merged_hidden_states = ops.interpolate(hidden_states, size=(new_h, new_w), mode=method)
    merged_hidden_states = merged_hidden_states.permute(0, 2, 3, 1).reshape(batch_size, -1, channel)
    return merged_hidden_states


@ms.jit_class
class ToDoJointAttnProcessor(JointAttnProcessor):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        residual = hidden_states

        batch_size, channel, height, width = (None,) * 4
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        if attn.use_downsample and attn.layer_idx <= 11:
            hidden_states = downsample(hidden_states, attn.token_merge_factor, method=attn.token_merge_method)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = ops.cat([query, encoder_hidden_states_query_proj], axis=1)
        key = ops.cat([key, encoder_hidden_states_key_proj], axis=1)
        value = ops.cat([value, encoder_hidden_states_value_proj], axis=1)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        hidden_states = ops.operations.nn_ops.FlashAttentionScore(1, scale_value=attn.scale)(
            query.to(ms.float16), key.to(ms.float16), value.to(ms.float16), None, None, None, attention_mask
        )[3].to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

