import logging
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.ops as ops

from mindone.transformers import Phi3Model
from mindone.transformers.cache_utils import Cache, DynamicCache
from mindone.transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.getLogger(__name__)


class Phi3Transformer(Phi3Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Modified version to work with OmniGen's attention mask handling.
    """

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) != (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        # TODO: temporally disable Cache
        # return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            # return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                print(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and attention_mask.ndim == 3:
            dtype = inputs_embeds.dtype
            min_dtype = self.dtype_to_min(dtype)
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = ops.expand_dims(attention_mask, 1).astype(inputs_embeds.dtype)
        else:
            raise Exception("attention_mask parameter was unavailable or invalid")

        hidden_states = inputs_embeds

        # Store intermediates
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Decoder layers
        layer_idx = -1
        for decoder_layer in self.layers:
            layer_idx += 1

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final layer norm
        hidden_states = self.norm(hidden_states)
        # Add hidden states from final layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return hidden_states, next_cache, all_hidden_states, all_self_attns
