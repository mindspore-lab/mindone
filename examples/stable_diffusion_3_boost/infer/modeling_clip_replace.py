from typing import Optional, Tuple

import mindspore as ms
from mindspore import ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore


def clip_attention_construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        causal_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        src_len = key_states.shape[1]
        attn_mask = None
        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, q_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_mask = causal_attention_mask if attention_mask is None else None 

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_mask = attention_mask if causal_attention_mask is None else causal_attention_mask + attention_mask

        attn_mask = ops.cast(attn_mask, dtype=ms.bool_) if attn_mask is not None else None

        _, _, softmax_out, attn_output = FlashAttentionScore(
             self.num_heads,
             keep_prob=1.0,
             sparse_mode=0,
             scale_value=self.scale,
             input_layout='BSH'
        )(query_states, key_states, value_states, None, None, None, attn_mask, None)

        attn_weights_reshaped = None if not output_attentions else softmax_out

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped
