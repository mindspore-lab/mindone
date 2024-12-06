from typing import Literal, Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ..common_layer import Embedding
from .layer import MistralAttention, MistralFlashAttention, MistralMLP, MistralRMSNorm

MISTRAL_ATTENTION_CLASSES = {
    "eager": MistralAttention,
    "flash_attention": MistralFlashAttention,
}


class MistralDecoderLayer(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 32768,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[attn_implementation](
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            dtype=dtype,
        )

        self.mlp = MistralMLP(
            intermediate_size=intermediate_size, hidden_size=hidden_size, hidden_act=hidden_act, dtype=dtype
        )
        self.input_layernorm = MistralRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = MistralRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_cache: Optional[Tensor] = None,
        past_value_cache: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, key_cache, value_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            return_key_value_cache=return_key_value_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, key_cache, value_cache


class MistralModel(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        vocab_size: int = 32064,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        pad_token_id: Optional[int] = None,
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        dtype: ms.dtype = ms.float32,
    ) -> None:
        super().__init__()
        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size
        self.attn_implementation = attn_implementation

        self.embed_tokens = Embedding(vocab_size, hidden_size, padding_idx=self.padding_idx, dtype=dtype)
        self.layers = nn.CellList(
            [
                MistralDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    rms_norm_eps=rms_norm_eps,
                    max_position_embeddings=max_position_embeddings,
                    rope_theta=rope_theta,
                    attention_dropout=attention_dropout,
                    hidden_act=hidden_act,
                    attn_implementation=attn_implementation,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = MistralRMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def get_input_embeddings(self) -> nn.Cell:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Cell) -> None:
        self.embed_tokens = value

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_cache_list: Optional[Tensor] = None,
        past_value_cache_list: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_cache_list.shape[-2] if past_key_cache_list is not None else 0
        cache_position = ops.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], dtype=ms.int32)
        if position_ids is None:
            position_ids = ops.unsqueeze(cache_position, 0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        hidden_states = inputs_embeds

        if return_key_value_cache:
            key_cache_list, value_cache_list = [], []
        else:
            key_cache_list, value_cache_list = None, None

        for i, decoder_layer in enumerate(self.layers):
            if past_key_cache_list is not None and past_value_cache_list is not None:
                past_key_cache, past_value_cache = past_key_cache_list[i], past_value_cache_list[i]
            else:
                past_key_cache, past_value_cache = None, None

            hidden_states, key_cache, value_cache = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_cache=past_key_cache,
                past_value_cache=past_value_cache,
                return_key_value_cache=return_key_value_cache,
            )

            if return_key_value_cache:
                key_cache_list.append(key_cache)
                value_cache_list.append(value_cache)

        hidden_states = self.norm(hidden_states)

        if return_key_value_cache:
            key_cache_list = ops.stack(key_cache_list)
            value_cache_list = ops.stack(value_cache_list)

        return hidden_states, key_cache_list, value_cache_list

    def _update_causal_mask(self, attention_mask: Tensor, input_tensor: Tensor, cache_position: Tensor) -> Tensor:
        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        target_length = attention_mask.shape[-1]

        if len(attention_mask.shape) == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            fill_value = -ms.numpy.inf if self.attn_implementation == "eager" else 1.0
            causal_mask = ops.full((sequence_length, target_length), fill_value=fill_value, dtype=dtype)
            exclude_mask = ops.arange(target_length) > cache_position.reshape(-1, 1)
            causal_mask = ops.masked_fill(causal_mask, ~exclude_mask, Tensor(0, dtype=dtype))
            causal_mask = ops.broadcast_to(causal_mask[None, None, :, :], (input_tensor.shape[0], 1, -1, -1))
            if len(attention_mask.shape) == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = ops.masked_fill(
                    causal_mask[:, :, :, :mask_length], padding_mask, Tensor(fill_value, dtype=dtype)
                )

        return causal_mask


class MistralForCausalLM(nn.Cell):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 32,
        num_hidden_layers: int = 32,
        num_key_value_heads: int = 8,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        vocab_size: int = 32000,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        pad_token_id: Optional[int] = None,
        attn_implementation: Literal["eager", "flash_attention"] = "eager",
        dtype: ms.dtype = ms.float32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = MistralModel(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            vocab_size=vocab_size,
            attention_dropout=attention_dropout,
            hidden_act=hidden_act,
            pad_token_id=pad_token_id,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        self.vocab_size = vocab_size
        self.lm_head = nn.Dense(hidden_size, vocab_size, has_bias=False, dtype=dtype)

    def get_input_embeddings(self) -> nn.Cell:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Cell) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Cell:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Cell) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Cell) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Cell:
        return self.model

    @ms.jit
    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_cache_list: Optional[Tensor] = None,
        past_value_cache_list: Optional[Tensor] = None,
        return_key_value_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states, key_cache_list, value_cache_list = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_cache_list=past_key_cache_list,
            past_value_cache_list=past_value_cache_list,
            return_key_value_cache=return_key_value_cache,
        )
        logits = self.lm_head(hidden_states).to(ms.float32)
        return logits, key_cache_list, value_cache_list
