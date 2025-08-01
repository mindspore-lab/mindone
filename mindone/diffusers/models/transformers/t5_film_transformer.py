# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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
from typing import Optional, Tuple

import mindspore as ms
from mindspore import mint, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..attention_processor import Attention
from ..embeddings import get_timestep_embedding
from ..modeling_utils import ModelMixin


class T5FilmDecoder(ModelMixin, ConfigMixin):
    r"""
    T5 style decoder with FiLM conditioning.

    Args:
        input_dims (`int`, *optional*, defaults to `128`):
            The number of input dimensions.
        targets_length (`int`, *optional*, defaults to `256`):
            The length of the targets.
        d_model (`int`, *optional*, defaults to `768`):
            Size of the input hidden states.
        num_layers (`int`, *optional*, defaults to `12`):
            The number of `DecoderLayer`'s to use.
        num_heads (`int`, *optional*, defaults to `12`):
            The number of attention heads to use.
        d_kv (`int`, *optional*, defaults to `64`):
            Size of the key-value projection vectors.
        d_ff (`int`, *optional*, defaults to `2048`):
            The number of dimensions in the intermediate feed-forward layer of `DecoderLayer`'s.
        dropout_rate (`float`, *optional*, defaults to `0.1`):
            Dropout probability.
    """

    @register_to_config
    def __init__(
        self,
        input_dims: int = 128,
        targets_length: int = 256,
        max_decoder_noise_time: float = 2000.0,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_kv: int = 64,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.conditioning_emb = nn.SequentialCell(
            mint.nn.Linear(d_model, d_model * 4, bias=False),
            mint.nn.SiLU(),
            mint.nn.Linear(d_model * 4, d_model * 4, bias=False),
            mint.nn.SiLU(),
        )

        self.position_encoding = mint.nn.Embedding(targets_length, d_model)
        self.position_encoding.weight.requires_grad = False

        self.position_encoding = mint.nn.Embedding(targets_length, d_model)
        self.position_encoding.weight.requires_grad = False

        self.continuous_inputs_projection = mint.nn.Linear(input_dims, d_model, bias=False)

        self.dropout = mint.nn.Dropout(p=dropout_rate)

        self.decoders = []
        for lyr_num in range(num_layers):
            # FiLM conditional T5 decoder
            lyr = DecoderLayer(d_model=d_model, d_kv=d_kv, num_heads=num_heads, d_ff=d_ff, dropout_rate=dropout_rate)
            self.decoders.append(lyr)
        self.decoders = nn.CellList(self.decoders)

        self.decoder_norm = T5LayerNorm(d_model)

        self.post_dropout = mint.nn.Dropout(p=dropout_rate)
        self.spec_out = mint.nn.Linear(d_model, input_dims, bias=False)

    def encoder_decoder_mask(self, query_input: ms.Tensor, key_input: ms.Tensor) -> ms.Tensor:
        mask = mint.mul(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
        return mask.unsqueeze(-3)

    def construct(self, encodings_and_masks, decoder_input_tokens, decoder_noise_time):
        batch, _, _ = decoder_input_tokens.shape
        assert decoder_noise_time.shape == (batch,)

        # decoder_noise_time is in [0, 1), so rescale to expected timing range.
        time_steps = get_timestep_embedding(
            decoder_noise_time * self.config["max_decoder_noise_time"],
            embedding_dim=self.config["d_model"],
            max_period=self.config["max_decoder_noise_time"],
        ).to(dtype=self.dtype)

        conditioning_emb = mint.unsqueeze(self.conditioning_emb(time_steps), 1)

        assert conditioning_emb.shape == (batch, 1, self.config["d_model"] * 4)

        seq_length = decoder_input_tokens.shape[1]

        # If we want to use relative positions for audio context, we can just offset
        # this sequence by the length of encodings_and_masks.
        decoder_positions = mint.broadcast_to(mint.arange(seq_length), (batch, seq_length))

        position_encodings = self.position_encoding(decoder_positions)

        inputs = self.continuous_inputs_projection(decoder_input_tokens)
        inputs += position_encodings
        y = self.dropout(inputs)

        # decoder: No padding present.
        decoder_mask = mint.ones(decoder_input_tokens.shape[:2], dtype=inputs.dtype)

        # Translate encoding masks to encoder-decoder masks.
        encodings_and_encdec_masks = [(x, self.encoder_decoder_mask(decoder_mask, y)) for x, y in encodings_and_masks]

        # cross attend style: concat encodings
        encoded = mint.cat([x[0] for x in encodings_and_encdec_masks], dim=1)
        encoder_decoder_mask = mint.cat([x[1] for x in encodings_and_encdec_masks], dim=-1)

        for lyr in self.decoders:
            y = lyr(
                y,
                conditioning_emb=conditioning_emb,
                encoder_hidden_states=encoded,
                encoder_attention_mask=encoder_decoder_mask,
            )[0]

        y = self.decoder_norm(y)
        y = self.post_dropout(y)

        spec_out = self.spec_out(y)
        return spec_out


class DecoderLayer(nn.Cell):
    r"""
    T5 decoder layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`, *optional*, defaults to `1e-6`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(
        self, d_model: int, d_kv: int, num_heads: int, d_ff: int, dropout_rate: float, layer_norm_epsilon: float = 1e-6
    ):
        super().__init__()
        layers = []

        # cond self attention: layer 0
        layers.append(
            T5LayerSelfAttentionCond(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate)
        )

        # cross attention: layer 1
        layers.append(
            T5LayerCrossAttention(
                d_model=d_model,
                d_kv=d_kv,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                layer_norm_epsilon=layer_norm_epsilon,
            )
        )

        # Film Cond MLP + dropout: last layer
        layers.append(
            T5LayerFFCond(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, layer_norm_epsilon=layer_norm_epsilon)
        )

        self.layer = nn.CellList(layers)

    def construct(
        self,
        hidden_states: ms.Tensor,
        conditioning_emb: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        encoder_decoder_position_bias=None,
    ) -> Tuple[ms.Tensor]:
        hidden_states = self.layer[0](
            hidden_states,
            conditioning_emb=conditioning_emb,
            attention_mask=attention_mask,
        )

        if encoder_hidden_states is not None:
            encoder_extended_attention_mask = mint.where(
                encoder_attention_mask > 0, ms.tensor(0.0), ms.tensor(-1e10)
            ).to(encoder_hidden_states.dtype)

            hidden_states = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_extended_attention_mask,
            )

        # Apply Film Conditional Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, conditioning_emb)

        return (hidden_states,)


class T5LayerSelfAttentionCond(nn.Cell):
    r"""
    T5 style self-attention layer with conditioning.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        dropout_rate (`float`):
            Dropout probability.
    """

    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.layer_norm = T5LayerNorm(d_model)
        self.FiLMLayer = T5FiLMLayer(in_features=d_model * 4, out_features=d_model)
        self.attention = Attention(query_dim=d_model, heads=num_heads, dim_head=d_kv, out_bias=False, scale_qk=False)
        self.dropout = mint.nn.Dropout(p=dropout_rate)

    def construct(
        self,
        hidden_states: ms.Tensor,
        conditioning_emb: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        # pre_self_attention_layer_norm
        normed_hidden_states = self.layer_norm(hidden_states)

        if conditioning_emb is not None:
            normed_hidden_states = self.FiLMLayer(normed_hidden_states, conditioning_emb)

        # Self-attention block
        attention_output = self.attention(normed_hidden_states)

        hidden_states = hidden_states + self.dropout(attention_output)

        return hidden_states


class T5LayerCrossAttention(nn.Cell):
    r"""
    T5 style cross-attention layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_kv (`int`):
            Size of the key-value projection vectors.
        num_heads (`int`):
            Number of attention heads.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout_rate: float, layer_norm_epsilon: float):
        super().__init__()
        self.attention = Attention(query_dim=d_model, heads=num_heads, dim_head=d_kv, out_bias=False, scale_qk=False)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = mint.nn.Dropout(p=dropout_rate)

    def construct(
        self,
        hidden_states: ms.Tensor,
        key_value_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.attention(
            normed_hidden_states,
            encoder_hidden_states=key_value_states,
            attention_mask=attention_mask.squeeze(1),
        )
        layer_output = hidden_states + self.dropout(attention_output)
        return layer_output


class T5LayerFFCond(nn.Cell):
    r"""
    T5 style feed-forward conditional layer.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
        layer_norm_epsilon (`float`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, layer_norm_epsilon: float):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate)
        self.film = T5FiLMLayer(in_features=d_model * 4, out_features=d_model)
        self.layer_norm = T5LayerNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = mint.nn.Dropout(p=dropout_rate)

    def construct(self, hidden_states: ms.Tensor, conditioning_emb: Optional[ms.Tensor] = None) -> ms.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        if conditioning_emb is not None:
            forwarded_states = self.film(forwarded_states, conditioning_emb)

        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5DenseGatedActDense(nn.Cell):
    r"""
    T5 style feed-forward layer with gated activations and dropout.

    Args:
        d_model (`int`):
            Size of the input hidden states.
        d_ff (`int`):
            Size of the intermediate feed-forward layer.
        dropout_rate (`float`):
            Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.wi_0 = mint.nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = mint.nn.Linear(d_model, d_ff, bias=False)
        self.wo = mint.nn.Linear(d_ff, d_model, bias=False)
        self.dropout = mint.nn.Dropout(p=dropout_rate)
        self.act = NewGELUActivation()

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerNorm(nn.Cell):
    r"""
    T5 style layer normalization module.

    Args:
        hidden_size (`int`):
            Size of the input hidden states.
        eps (`float`, `optional`, defaults to `1e-6`):
            A small value used for numerical stability to avoid dividing by zero.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = ms.Parameter(mint.ones(hidden_size), name="weight")
        self.variance_epsilon = eps

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://huggingface.co/papers/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(ms.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * mint.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [ms.float16, ms.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class NewGELUActivation(nn.Cell):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        # Magic number 0.797885 comes from math.sqrt(2.0 / math.pi) as float32
        return 0.5 * input * (1.0 + mint.tanh(0.797885 * (input + 0.044715 * mint.pow(input, 3.0))))


class T5FiLMLayer(nn.Cell):
    """
    T5 style FiLM Layer.

    Args:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.scale_bias = mint.nn.Linear(in_features, out_features * 2, bias=False)

    def construct(self, x: ms.Tensor, conditioning_emb: ms.Tensor) -> ms.Tensor:
        emb = self.scale_bias(conditioning_emb)
        scale, shift = mint.chunk(emb, 2, -1)
        x = x * (1 + scale) + shift
        return x
