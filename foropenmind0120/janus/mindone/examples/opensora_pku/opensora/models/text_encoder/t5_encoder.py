import copy
import logging

import mint
import numpy as np
from transformers.models.t5.configuration_t5 import T5Config

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindone.transformers.activations import ACT2FN
from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel

__all__ = ["T5EncoderModel"]
logger = logging.getLogger(__name__)


def arange(start=0, end=None, step=1, *, dtype=None):
    """inner arange"""
    res = ms.Tensor(np.arange(start, end, step))
    if dtype is not None:
        res = res.astype(dtype)
    return res


class T5LayerNorm(nn.Cell):
    """T5LayerNorm"""

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = ms.Parameter(ops.ones(hidden_size, ms.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        variance = hidden_states.astype(ms.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states / ops.sqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [ms.float16]:
            hidden_states = hidden_states.astype(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseActDense(nn.Cell):
    """T5DenseActDense"""

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.wo.weight.dtype not in (hidden_states.dtype, ms.int8):
            hidden_states = hidden_states.astype(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Cell):
    """T5DenseGatedActDense"""

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wi_1 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        if self.wo.weight.dtype not in (hidden_states.dtype, ms.int8):
            hidden_states = hidden_states.astype(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Cell):
    """T5LayerFF"""

    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Cell):
    """T5Attention"""

    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.output_attentions = False
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.k = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.v = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.o = nn.Dense(self.inner_dim, self.d_model, has_bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(ms.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = 0 - ops.minimum(relative_position, ops.zeros(relative_position.shape)).astype(ms.int64)
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = (
            max_exact
            + ops.log(relative_position.astype(ms.float32) / max_exact)
            / ms.numpy.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(ms.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large,
            ops.fill(ms.int64, relative_position_if_large.shape, num_buckets - 1),
        )
        # relative_buckets += ops.where(is_small, relative_position\
        # , relative_position_if_large) # mindspore 2.0
        relative_buckets += ops.select(
            is_small.astype(ms.bool_), relative_position, relative_position_if_large
        )  # mindspore 1.10
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = arange(query_length, dtype=ms.int64)[:, None]
        memory_position = arange(key_length, dtype=ms.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.transpose([2, 0, 1]).expand_dims(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def construct(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).swapaxes(1, 2)

        def unshape(states):
            """reshape"""
            return states.swapaxes(1, 2).view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = mint.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = ops.matmul(
            query_states, key_states.swapaxes(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros((1, self.n_heads, real_seq_length, key_length), scores.dtype)
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = ops.softmax(scores.astype(ms.float32), axis=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        if self.training:
            attn_weights = ops.dropout(attn_weights, p=self.dropout)  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(ops.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and self.use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Cell):
    """T5LayerSelfAttention"""

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.use_cache = config.use_cache

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Cell):
    """T5LayerCrossAttention"""

    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.use_cache = config.use_cache

    def construct(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Cell):
    """T5Block"""

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.CellList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))
        self.use_cache = config.use_cache

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == ms.float16 and ops.isinf(hidden_states).any():
            clamp_value = np.finfo(np.float16).max - 1000.0
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == ms.float16 and ops.isinf(hidden_states).any():
                clamp_value = np.finfo(np.float16).max - 1000.0
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == ms.float16 and ops.isinf(hidden_states).any():
            clamp_value = np.finfo(np.float16).max - 1000.0
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if self.use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs
        # hidden-states, present_key_value_states, (self-attention position bias),
        # (self-attention weights), (cross-attention position bias),(cross-attention weights)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    base_model_prefix = "transformer"
    # pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    # TODO
    def get_input_embeddings(self):
        pass

    # TODO
    def get_position_embeddings(self):
        pass

    # TODO
    def resize_position_embeddings(self):
        pass

    # TODO
    def set_input_embeddings(self):
        pass

    # TODO
    def post_init(self):
        pass

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = ops.zeros(input_ids.shape, input_ids.dtype)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].copy()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = ops.masked_fill(shifted_input_ids, shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    """T5Stack"""

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.num_layers = config.num_layers

        self.block = nn.CellList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.use_cache = config.use_cache

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if self.use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, mask_seq_length), ms.float32)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = ops.ones((batch_size, encoder_seq_length), ms.int64)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.num_layers)
        present_key_value_states = []
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), \
            # (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if self.use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3]
            # append next layer key value states
            if self.use_cache:
                present_key_value_states.append(present_key_value_state)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
            ]
            if v is not None and len(v) > 0
        )


class T5EncoderModel(T5PreTrainedModel):
    """T5EncoderModel"""

    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        """get encoder"""
        return self.encoder

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
        )

        return encoder_outputs
