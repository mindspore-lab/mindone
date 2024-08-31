# coding=utf-8
# Copyright 2023 The Fairseq Authors, Microsoft Research, and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SpeechT5 model."""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Parameter
from mindnlp.modules.weight_norm import weight_norm, remove_weight_norm

from mindnlp.transformers.activations import ACT2FN
from mindnlp.transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from mindnlp.transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSpectrogramOutput,
)
from mindnlp.transformers.modeling_utils import PreTrainedModel
from mindnlp.utils import logging
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 1

# General docstring
_CONFIG_FOR_DOC = "SpeechT5Config"


SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/speecht5_asr",
    "microsoft/speecht5_tts",
    "microsoft/speecht5_vc",
    # See all SpeechT5 models at https://huggingface.co/models?filter=speecht5
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
# def shift_tokens_right(input_ids: ms.Tensor, pad_token_id: int, decoder_start_token_id: int):
#     """
#     Shift input ids one token to the right.
#     """
#     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
#     shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
#     shifted_input_ids[:, 0] = decoder_start_token_id
#
#     if pad_token_id is None:
#         raise ValueError("self.model.config.pad_token_id has to be defined.")
#     # replace possible -100 values in labels by `pad_token_id`
#     shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
#
#     return shifted_input_ids
#
#
# def shift_spectrograms_right(input_values: ms.Tensor, reduction_factor: int = 1):
#     """
#     Shift input spectrograms one timestep to the right. Also applies the reduction factor to the sequence length.
#     """
#     # thin out frames for reduction factor
#     if reduction_factor > 1:
#         input_values = input_values[:, reduction_factor - 1 :: reduction_factor]
#
#     shifted_input_values = input_values.new_zeros(input_values.shape)
#     shifted_input_values[:, 1:] = input_values[:, :-1].clone()
#
#     # replace possible -100 values in labels by zeros
#     shifted_input_values.masked_fill_(shifted_input_values == -100.0, 0.0)
#
#     return shifted_input_values
#
#
# # Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
# def _compute_mask_indices(
#     shape: Tuple[int, int],
#     mask_prob: float,
#     mask_length: int,
#     attention_mask: Optional[ms.Tensor] = None,
#     min_masks: int = 0,
# ) -> np.ndarray:
#     """
#     Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
#     ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
#     CPU as part of the preprocessing during training.
#
#     Args:
#         shape: The shape for which to compute masks. This should be of a tuple of size 2 where
#                the first element is the batch size and the second element is the length of the axis to span.
#         mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
#                     independently generated mask spans of length `mask_length` is computed by
#                     `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
#                     actual percentage will be smaller.
#         mask_length: size of the mask
#         min_masks: minimum number of masked spans
#         attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
#                         each batch dimension.
#     """
#     batch_size, sequence_length = shape
#
#     if mask_length < 1:
#         raise ValueError("`mask_length` has to be bigger than 0.")
#
#     if mask_length > sequence_length:
#         raise ValueError(
#             f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
#             f" and `sequence_length`: {sequence_length}`"
#         )
#
#     # epsilon is used for probabilistic rounding
#     epsilon = np.random.rand(1).item()
#
#     def compute_num_masked_span(input_length):
#         """Given input length, compute how many spans should be masked"""
#         num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
#         num_masked_span = max(num_masked_span, min_masks)
#
#         # make sure num masked span <= sequence_length
#         if num_masked_span * mask_length > sequence_length:
#             num_masked_span = sequence_length // mask_length
#
#         # make sure num_masked span is also <= input_length - (mask_length - 1)
#         if input_length - (mask_length - 1) < num_masked_span:
#             num_masked_span = max(input_length - (mask_length - 1), 0)
#
#         return num_masked_span
#
#     # compute number of masked spans in batch
#     input_lengths = (
#         attention_mask.sum(-1).detach().tolist()
#         if attention_mask is not None
#         else [sequence_length for _ in range(batch_size)]
#     )
#
#     # SpecAugment mask to fill
#     spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
#     spec_aug_mask_idxs = []
#
#     max_num_masked_span = compute_num_masked_span(sequence_length)
#
#     if max_num_masked_span == 0:
#         return spec_aug_mask
#
#     for input_length in input_lengths:
#         # compute num of masked spans for this input
#         num_masked_span = compute_num_masked_span(input_length)
#
#         # get random indices to mask
#         spec_aug_mask_idx = np.random.choice(
#             np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
#         )
#
#         # pick first sampled index that will serve as a dummy index to pad vector
#         # to ensure same dimension for all batches due to probabilistic rounding
#         # Picking first sample just pads those vectors twice.
#         if len(spec_aug_mask_idx) == 0:
#             # this case can only happen if `input_length` is strictly smaller then
#             # `sequence_length` in which case the last token has to be a padding
#             # token which we can use as a dummy mask id
#             dummy_mask_idx = sequence_length - 1
#         else:
#             dummy_mask_idx = spec_aug_mask_idx[0]
#
#         spec_aug_mask_idx = np.concatenate(
#             [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
#         )
#         spec_aug_mask_idxs.append(spec_aug_mask_idx)
#
#     spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)
#
#     # expand masked indices to masked spans
#     spec_aug_mask_idxs = np.broadcast_to(
#         spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
#     )
#     spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)
#
#     # add offset to the starting indexes so that indexes now create a span
#     offsets = np.arange(mask_length)[None, None, :]
#     offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
#         batch_size, max_num_masked_span * mask_length
#     )
#     spec_aug_mask_idxs = spec_aug_mask_idxs + offsets
#
#     # ensure that we cannot have indices larger than sequence_length
#     if spec_aug_mask_idxs.max() > sequence_length - 1:
#         spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1
#
#     # scatter indices to mask
#     np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)
#
#     return spec_aug_mask
#
#
# # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->SpeechT5
# class SpeechT5NoLayerNormConvLayer(nn.Cell):
#     def __init__(self, config, layer_id=0):
#         super().__init__()
#         self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
#         self.out_conv_dim = config.conv_dim[layer_id]
#
#         self.conv = nn.Conv1d(
#             self.in_conv_dim,
#             self.out_conv_dim,
#             kernel_size=config.conv_kernel[layer_id],
#             stride=config.conv_stride[layer_id],
#             bias=config.conv_bias,
#         )
#         self.activation = ACT2FN[config.feat_extract_activation]
#
#     def construct(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         return hidden_states
#
#
# # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->SpeechT5
# class SpeechT5LayerNormConvLayer(nn.Cell):
#     def __init__(self, config, layer_id=0):
#         super().__init__()
#         self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
#         self.out_conv_dim = config.conv_dim[layer_id]
#
#         self.conv = nn.Conv1d(
#             self.in_conv_dim,
#             self.out_conv_dim,
#             kernel_size=config.conv_kernel[layer_id],
#             stride=config.conv_stride[layer_id],
#             bias=config.conv_bias,
#         )
#         self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
#         self.activation = ACT2FN[config.feat_extract_activation]
#
#     def construct(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#
#         hidden_states = hidden_states.transpose(-2, -1)
#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = hidden_states.transpose(-2, -1)
#
#         hidden_states = self.activation(hidden_states)
#         return hidden_states
#
#
# # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->SpeechT5
# class SpeechT5GroupNormConvLayer(nn.Cell):
#     def __init__(self, config, layer_id=0):
#         super().__init__()
#         self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
#         self.out_conv_dim = config.conv_dim[layer_id]
#
#         self.conv = nn.Conv1d(
#             self.in_conv_dim,
#             self.out_conv_dim,
#             kernel_size=config.conv_kernel[layer_id],
#             stride=config.conv_stride[layer_id],
#             bias=config.conv_bias,
#         )
#         self.activation = ACT2FN[config.feat_extract_activation]
#
#         self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)
#
#     def construct(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         return hidden_states
#

# Copied from transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->SpeechT5
# class SpeechT5SinusoidalPositionalEmbedding(nn.Cell):
#     """This module produces sinusoidal positional embeddings of any length."""
#
#     def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
#         super().__init__()
#         self.offset = 2
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx
#         self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)
#
#     def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
#         emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
#         if hasattr(self, "weights"):
#             # in forward put the weights on the correct dtype and device of the param
#             emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)
#
#         self.weights = Parameter(emb_weights)
#         self.weights.requires_grad = False
#         self.weights.detach_()
#
#     @staticmethod
#     def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
#         """
#         Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
#         description in Section 3.5 of "Attention Is All You Need".
#         """
#         half_dim = embedding_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = ops.exp(ops.arange(half_dim, dtype=ms.int64).float() * -emb)
#         emb = ops.arange(num_embeddings, dtype=ms.int64).float().unsqueeze(1) * emb.unsqueeze(0)
#         emb = ops.cat([ops.sin(emb), ops.cos(emb)], axis=1).view(num_embeddings, -1)
#         if embedding_dim % 2 == 1:
#             # zero pad
#             emb = ops.cat([emb, ops.zeros(num_embeddings, 1)], axis=1)
#         if padding_idx is not None:
#             emb[padding_idx, :] = 0
#         return emb.to(ms.float32)
#
#     def construct(self, input_ids: ms.Tensor, past_key_values_length: int = 0):
#         bsz, seq_len = input_ids.size()
#         # Create the position ids from the input token ids. Any padded tokens remain padded.
#         position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
#             input_ids.device
#         )
#
#         # expand embeddings if needed
#         max_pos = self.padding_idx + 1 + seq_len
#         if max_pos > self.weights.size(0):
#             self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)
#
#         return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
#
#     def create_position_ids_from_input_ids(
#         self, input_ids: ms.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
#     ):
#         """
#         Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
#         symbols are ignored. This is modified from fairseq's `utils.make_positions`.
#
#         Args:
#             x: torch.Tensor x:
#         Returns: torch.Tensor
#         """
#         # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
#         mask = input_ids.ne(padding_idx).int()
#         incremental_indices = (ops.cumsum(mask, axis=1).type_as(mask) + past_key_values_length) * mask
#         return incremental_indices.long() + padding_idx
#
# # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding with Wav2Vec2->SpeechT5
# class SpeechT5PositionalConvEmbedding(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.conv = nn.Conv1d(
#             config.hidden_size,
#             config.hidden_size,
#             kernel_size=config.num_conv_pos_embeddings,
#             padding=config.num_conv_pos_embeddings // 2,
#             groups=config.num_conv_pos_embedding_groups,
#         )
#
#
#         if is_deepspeed_zero3_enabled():
#             import deepspeed
#
#             with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
#                 self.conv = weight_norm(self.conv, name="weight", dim=2)
#             deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
#             deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
#         else:
#             self.conv = weight_norm(self.conv, name="weight", dim=2)
#
#         self.padding = SpeechT5SamePadLayer(config.num_conv_pos_embeddings)
#         self.activation = ACT2FN[config.feat_extract_activation]
#
#     def construct(self, hidden_states):
#         hidden_states = hidden_states.transpose(1, 2)
#
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.padding(hidden_states)
#         hidden_states = self.activation(hidden_states)
#
#         hidden_states = hidden_states.transpose(1, 2)
#         return hidden_states
#
#
# class SpeechT5ScaledPositionalEncoding(nn.Cell):
#     """
#     Scaled positional encoding, see §3.2 in https://arxiv.org/abs/1809.08895
#     """
#
#     def __init__(self, dropout, dim, max_len=5000):
#         pe = ops.zeros(max_len, dim)
#         position = ops.arange(0, max_len).unsqueeze(1)
#         div_term = ops.exp((ops.arange(0, dim, 2, dtype=ms.int64).float() * -(math.log(10000.0) / dim)))
#         pe[:, 0::2] = ops.sin(position.float() * div_term)
#         pe[:, 1::2] = ops.cos(position.float() * div_term)
#         pe = pe.unsqueeze(0)
#         super().__init__()
#         self.register_buffer("pe", pe, persistent=False)
#         self.dropout = nn.Dropout(p=dropout)
#         self.dim = dim
#         self.alpha = Parameter(ms.tensor(1.0))
#
#     def construct(self, emb):
#         emb = emb + self.alpha * self.pe[:, : emb.size(1)]
#         emb = self.dropout(emb)
#         return emb
#
#
# class SpeechT5RelativePositionalEncoding(nn.Cell):
#     def __init__(self, dim, max_length=1000):
#         super().__init__()
#         self.dim = dim
#         self.max_length = max_length
#         self.pe_k = nn.Embedding(2 * max_length, dim)
#
#     def construct(self, hidden_states):
#         seq_len = hidden_states.shape[1]
#         pos_seq = ops.arange(0, seq_len).long().to(hidden_states.device)
#         pos_seq = pos_seq[:, None] - pos_seq[None, :]
#
#         pos_seq[pos_seq < -self.max_length] = -self.max_length
#         pos_seq[pos_seq >= self.max_length] = self.max_length - 1
#         pos_seq = pos_seq + self.max_length
#
#         return self.pe_k(pos_seq)
#
#
# # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer with Wav2Vec2->SpeechT5
# class SpeechT5SamePadLayer(nn.Cell):
#     def __init__(self, num_conv_pos_embeddings):
#         super().__init__()
#         self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0
#
#     def construct(self, hidden_states):
#         if self.num_pad_remove > 0:
#             hidden_states = hidden_states[:, :, : -self.num_pad_remove]
#         return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder with Wav2Vec2->SpeechT5
# class SpeechT5FeatureEncoder(nn.Cell):
#     """Construct the features from raw audio waveform"""
#
#     def __init__(self, config):
#         super().__init__()
#
#         if config.feat_extract_norm == "group":
#             conv_layers = [SpeechT5GroupNormConvLayer(config, layer_id=0)] + [
#                 SpeechT5NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
#             ]
#         elif config.feat_extract_norm == "layer":
#             conv_layers = [
#                 SpeechT5LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
#             ]
#         else:
#             raise ValueError(
#                 f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
#             )
#         self.conv_layers = nn.ModuleList(conv_layers)
#         self.gradient_checkpointing = False
#         self._requires_grad = True
#
#     def _freeze_parameters(self):
#         for param in self.parameters():
#             param.requires_grad = False
#         self._requires_grad = False
#
#     def construct(self, input_values):
#         hidden_states = input_values[:, None]
#
#         # make sure hidden_states require grad for gradient_checkpointing
#         if self._requires_grad and self.training:
#             hidden_states.requires_grad = True
#
#         for conv_layer in self.conv_layers:
#             if self._requires_grad and self.gradient_checkpointing and self.training:
#                 hidden_states = self._gradient_checkpointing_func(
#                     conv_layer.__call__,
#                     hidden_states,
#                 )
#             else:
#                 hidden_states = conv_layer(hidden_states)
#
#         return hidden_states

#
# # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection with Wav2Vec2->SpeechT5
# class SpeechT5FeatureProjection(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(config.conv_dim[-1], epsilon=config.layer_norm_eps)
#         self.projection = nn.Dense(config.conv_dim[-1], config.hidden_size)
#         self.dropout = nn.Dropout(config.feat_proj_dropout)
#
#     def construct(self, hidden_states):
#         # non-projected hidden states are needed for quantization
#         norm_hidden_states = self.layer_norm(hidden_states)
#         hidden_states = self.projection(norm_hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         return hidden_states, norm_hidden_states
#
#
# class SpeechT5SpeechEncoderPrenet(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.feature_encoder = SpeechT5FeatureEncoder(config)
#         self.feature_projection = SpeechT5FeatureProjection(config)
#
#         # model only needs masking vector if mask prob is > 0.0
#         if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
#             self.masked_spec_embed = Parameter(ms.Tensor(config.hidden_size).uniform_())
#
#         self.pos_conv_embed = SpeechT5PositionalConvEmbedding(config)
#         self.pos_sinusoidal_embed = SpeechT5SinusoidalPositionalEmbedding(
#             config.max_speech_positions + config.pad_token_id + 1,
#             config.hidden_size,
#             config.pad_token_id,
#         )
#
#     def freeze_feature_encoder(self):
#         self.feature_encoder._freeze_parameters()
#
#     def construct(
#         self,
#         input_values: ms.Tensor,
#         attention_mask: Optional[ms.Tensor] = None,
#         mask_time_indices: Optional[ms.Tensor] = None,
#     ):
#         extract_features = self.feature_encoder(input_values)
#         extract_features = extract_features.transpose(1, 2)
#
#         if attention_mask is not None:
#             # compute reduced attention_mask corresponding to feature vectors
#             attention_mask = self._get_feature_vector_attention_mask(
#                 extract_features.shape[1],
#                 attention_mask,
#             )
#
#         hidden_states, extract_features = self.feature_projection(extract_features)
#         hidden_states = self._mask_hidden_states(
#             hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
#         )
#
#         positional_conv_embedding = self.pos_conv_embed(hidden_states)
#         hidden_states = hidden_states + positional_conv_embedding
#
#         if attention_mask is not None:
#             padding_mask = attention_mask.ne(1).long()
#         else:
#             padding_mask = ops.zeros(hidden_states.shape[:2], dtype=ms.int64)
#
#         positional_sinusoidal_embeddings = self.pos_sinusoidal_embed(padding_mask)
#         hidden_states = hidden_states + positional_sinusoidal_embeddings
#
#         return hidden_states, attention_mask
#
#     # Copied from transformers.models.unispeech.modeling_unispeech.UniSpeechPreTrainedModel._get_feature_vector_attention_mask
#     def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: ms.Tensor):
#         # Effectively attention_mask.sum(-1), but not inplace to be able to run
#         # on inference mode.
#         non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
#         output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(ms.int64)
#         batch_size = attention_mask.shape[0]
#
#         attention_mask = ops.zeros(
#             (batch_size, feature_vector_length), dtype=attention_mask.dtype
#         )
#         # these two operations makes sure that all values before the output lengths idxs are attended to
#         attention_mask[(ops.arange(attention_mask.shape[0]), output_lengths - 1)] = 1
#         attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
#         return attention_mask
#
#     # Copied from transformers.models.unispeech.modeling_unispeech.UniSpeechPreTrainedModel._get_feat_extract_output_lengths
#     def _get_feat_extract_output_lengths(self, input_lengths: Union[ms.Tensor, int]):
#         """
#         Computes the output length of the convolutional layers
#         """
#
#         def _conv_out_length(input_length, kernel_size, stride):
#             # 1D convolutional layer output length formula taken
#             # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
#             return ops.div(input_length - kernel_size, stride, rounding_mode="floor") + 1
#
#         for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
#             input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
#
#         return input_lengths
#
#     # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states
#     def _mask_hidden_states(
#         self,
#         hidden_states: ms.Tensor,
#         mask_time_indices: Optional[ms.Tensor] = None,
#         attention_mask: Optional[ms.Tensor] = None,
#     ):
#         """
#         Masks extracted features along time axis and/or along feature axis according to
#         [SpecAugment](https://arxiv.org/abs/1904.08779).
#         """
#
#         # `config.apply_spec_augment` can set masking to False
#         if not getattr(self.config, "apply_spec_augment", True):
#             return hidden_states
#
#         # generate indices & apply SpecAugment along time axis
#         batch_size, sequence_length, hidden_size = hidden_states.size()
#
#         if mask_time_indices is not None:
#             # apply SpecAugment along time axis with given mask_time_indices
#             hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
#         elif self.config.mask_time_prob > 0 and self.training:
#             mask_time_indices = _compute_mask_indices(
#                 (batch_size, sequence_length),
#                 mask_prob=self.config.mask_time_prob,
#                 mask_length=self.config.mask_time_length,
#                 attention_mask=attention_mask,
#                 min_masks=self.config.mask_time_min_masks,
#             )
#             mask_time_indices = ms.tensor(mask_time_indices, dtype=ms.bool_)
#             hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
#
#         if self.config.mask_feature_prob > 0 and self.training:
#             # generate indices & apply SpecAugment along feature axis
#             mask_feature_indices = _compute_mask_indices(
#                 (batch_size, hidden_size),
#                 mask_prob=self.config.mask_feature_prob,
#                 mask_length=self.config.mask_feature_length,
#                 min_masks=self.config.mask_feature_min_masks,
#             )
#             mask_feature_indices = ms.tensor(mask_feature_indices, dtype=ms.bool_)
#             mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
#             hidden_states[mask_feature_indices] = 0
#
#         return hidden_states
#
#
# class SpeechT5SpeechDecoderPrenet(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#
#         self.layers = nn.CellList(
#             [
#                 nn.Dense(
#                     config.num_mel_bins if i == 0 else config.speech_decoder_prenet_units,
#                     config.speech_decoder_prenet_units,
#                 )
#                 for i in range(config.speech_decoder_prenet_layers)
#             ]
#         )
#
#         self.final_layer = nn.Dense(config.speech_decoder_prenet_units, config.hidden_size)
#         self.encode_positions = SpeechT5ScaledPositionalEncoding(
#             config.positional_dropout,
#             config.hidden_size,
#             config.max_speech_positions,
#         )
#         self.speaker_embeds_layer = nn.Dense(config.speaker_embedding_dim + config.hidden_size, config.hidden_size)
#
#     def _consistent_dropout(self, inputs_embeds, p):
#         mask = ops.bernoulli(inputs_embeds[0], p=p)
#         all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
#         return ops.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)
#
#     def construct(
#         self,
#         input_values: ms.Tensor,
#         speaker_embeddings: Optional[ms.Tensor] = None,
#     ):
#         # Dropout is always applied, even when evaluating. See §2.2 in https://arxiv.org/abs/1712.05884.
#
#         inputs_embeds = input_values
#         for layer in self.layers:
#             inputs_embeds = ops.relu(layer(inputs_embeds))
#             inputs_embeds = self._consistent_dropout(inputs_embeds, self.config.speech_decoder_prenet_dropout)
#
#         inputs_embeds = self.final_layer(inputs_embeds)
#         inputs_embeds = self.encode_positions(inputs_embeds)
#
#         if speaker_embeddings is not None:
#             speaker_embeddings = ops.L2Normalize(speaker_embeddings)
#             speaker_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)
#             inputs_embeds = ops.cat([inputs_embeds, speaker_embeddings], axis=-1)
#             inputs_embeds = ops.relu(self.speaker_embeds_layer(inputs_embeds))
#
#         return inputs_embeds
#
#
# class SpeechT5BatchNormConvLayer(nn.Cell):
#     def __init__(self, config, layer_id=0):
#         super().__init__()
#
#         if layer_id == 0:
#             in_conv_dim = config.num_mel_bins
#         else:
#             in_conv_dim = config.speech_decoder_postnet_units
#
#         if layer_id == config.speech_decoder_postnet_layers - 1:
#             out_conv_dim = config.num_mel_bins
#         else:
#             out_conv_dim = config.speech_decoder_postnet_units
#
#         self.conv = nn.Conv1d(
#             in_conv_dim,
#             out_conv_dim,
#             kernel_size=config.speech_decoder_postnet_kernel,
#             stride=1,
#             padding=(config.speech_decoder_postnet_kernel - 1) // 2,
#             bias=False,
#         )
#         self.batch_norm = nn.BatchNorm1d(out_conv_dim)
#
#         if layer_id < config.speech_decoder_postnet_layers - 1:
#             self.activation = nn.Tanh()
#         else:
#             self.activation = None
#
#         self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)
#
#     def construct(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.batch_norm(hidden_states)
#         if self.activation is not None:
#             hidden_states = self.activation(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         return hidden_states
#
#
# class SpeechT5SpeechDecoderPostnet(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#
#         self.feat_out = nn.Dense(config.hidden_size, config.num_mel_bins * config.reduction_factor)
#         self.prob_out = nn.Dense(config.hidden_size, config.reduction_factor)
#
#         self.layers = nn.CellList(
#             [SpeechT5BatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
#         )
#
#     def construct(self, hidden_states: ms.Tensor):
#         outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
#         outputs_after_postnet = self.postnet(outputs_before_postnet)
#         logits = self.prob_out(hidden_states).view(hidden_states.size(0), -1)
#         return outputs_before_postnet, outputs_after_postnet, logits
#
#     def postnet(self, hidden_states: ms.Tensor):
#         layer_output = hidden_states.transpose(1, 2)
#         for layer in self.layers:
#             layer_output = layer(layer_output)
#         return hidden_states + layer_output.transpose(1, 2)
#
#
# class SpeechT5TextEncoderPrenet(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
#         self.encode_positions = SpeechT5ScaledPositionalEncoding(
#             config.positional_dropout,
#             config.hidden_size,
#             config.max_text_positions,
#         )
#
#     def get_input_embeddings(self):
#         return self.embed_tokens
#
#     def set_input_embeddings(self, value):
#         self.embed_tokens = value
#
#     def construct(self, input_ids: ms.Tensor):
#         inputs_embeds = self.embed_tokens(input_ids)
#         inputs_embeds = self.encode_positions(inputs_embeds)
#         return inputs_embeds
#
#
# class SpeechT5TextDecoderPrenet(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.dropout = nn.Dropout(config.positional_dropout)
#         self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
#
#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
#
#         self.embed_positions = SpeechT5SinusoidalPositionalEmbedding(
#             config.max_text_positions + config.pad_token_id + 1,
#             config.hidden_size,
#             config.pad_token_id,
#         )
#
#     def get_input_embeddings(self):
#         return self.embed_tokens
#
#     def set_input_embeddings(self, value):
#         self.embed_tokens = value
#
#     def construct(
#         self,
#         input_ids: ms.Tensor,
#         attention_mask: Optional[ms.Tensor] = None,
#         past_key_values: Optional[List[ms.Tensor]] = None,
#     ):
#         if input_ids is not None:
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
#         else:
#             raise ValueError("You have to specify `decoder_input_ids`")
#
#         past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
#         positions = self.embed_positions(input_ids, past_key_values_length)
#
#         inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
#         inputs_embeds += positions
#         inputs_embeds = self.dropout(inputs_embeds)
#
#         return inputs_embeds, attention_mask
#
#
# class SpeechT5TextDecoderPostnet(nn.Cell):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, bias=False)
#
#     def construct(self, hidden_states: ms.Tensor):
#         return self.lm_head(hidden_states)
#
#     def get_output_embeddings(self):
#         return self.lm_head
#
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings
#
#
# class SpeechT5Attention(nn.Cell):
#     """
#     Multi-headed attention from 'Attention Is All You Need' paper with relative position bias (see
#     https://aclanthology.org/N18-2074.pdf)
#     """
#
#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         is_decoder: bool = False,
#         bias: bool = True,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.head_dim = embed_dim // num_heads
#
#         if (self.head_dim * num_heads) != self.embed_dim:
#             raise ValueError(
#                 f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
#                 f" and `num_heads`: {num_heads})."
#             )
#         self.scaling = self.head_dim**-0.5
#         self.is_decoder = is_decoder
#
#         self.k_proj = nn.Dense(embed_dim, embed_dim, bias=bias)
#         self.v_proj = nn.Dense(embed_dim, embed_dim, bias=bias)
#         self.q_proj = nn.Dense(embed_dim, embed_dim, bias=bias)
#         self.out_proj = nn.Dense(embed_dim, embed_dim, bias=bias)
#
#     def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
#         return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
#
#     def construct(
#         self,
#         hidden_states: ms.Tensor,
#         key_value_states: Optional[ms.Tensor] = None,
#         past_key_value: Optional[Tuple[ms.Tensor]] = None,
#         attention_mask: Optional[ms.Tensor] = None,
#         layer_head_mask: Optional[ms.Tensor] = None,
#         position_bias: Optional[ms.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
#         """Input shape: Batch x Time x Channel"""
#
#         # if key_value_states are provided this layer is used as a cross-attention layer
#         # for the decoder
#         is_cross_attention = key_value_states is not None
#
#         bsz, tgt_len, _ = hidden_states.size()
#
#         # get query proj
#         query_states = self.q_proj(hidden_states) * self.scaling
#         # get key, value proj
#         if is_cross_attention and past_key_value is not None:
#             # reuse k,v, cross_attentions
#             key_states = past_key_value[0]
#             value_states = past_key_value[1]
#         elif is_cross_attention:
#             # cross_attentions
#             key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
#             value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
#         elif past_key_value is not None:
#             # reuse k, v, self_attention
#             key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
#             value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
#             key_states = ops.cat([past_key_value[0], key_states], axis=2)
#             value_states = ops.cat([past_key_value[1], value_states], axis=2)
#         else:
#             # self_attention
#             key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
#             value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
#
#         if self.is_decoder:
#             # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
#             # Further calls to cross_attention layer can then reuse all cross-attention
#             # key/value_states (first "if" case)
#             # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
#             # all previous decoder key/value_states. Further calls to uni-directional self-attention
#             # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
#             # if encoder bi-directional self-attention `past_key_value` is always `None`
#             past_key_value = (key_states, value_states)
#
#         proj_shape = (bsz * self.num_heads, -1, self.head_dim)
#         query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
#         key_states = key_states.view(*proj_shape)
#         value_states = value_states.view(*proj_shape)
#
#         src_len = key_states.size(1)
#         attn_weights = ops.bmm(query_states, key_states.transpose(1, 2))
#
#         if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
#             raise ValueError(
#                 f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
#                 f" {attn_weights.size()}"
#             )
#
#         # relative attention bias
#         if position_bias is not None:
#             reshape_q = query_states.contiguous().view(bsz * self.num_heads, -1, self.head_dim).transpose(0, 1)
#             rel_pos_bias = ops.matmul(reshape_q, position_bias.transpose(-2, -1))
#             rel_pos_bias = rel_pos_bias.transpose(0, 1).view(
#                 bsz * self.num_heads, position_bias.size(0), position_bias.size(1)
#             )
#             attn_weights += rel_pos_bias
#
#         if attention_mask is not None:
#             if attention_mask.size() != (bsz, 1, tgt_len, src_len):
#                 raise ValueError(
#                     f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
#                 )
#             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
#
#         attn_weights = ops.softmax(attn_weights, axis=-1)
#
#         if layer_head_mask is not None:
#             if layer_head_mask.size() != (self.num_heads,):
#                 raise ValueError(
#                     f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
#                     f" {layer_head_mask.size()}"
#                 )
#             attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
#
#         if output_attentions:
#             # this operation is a bit awkward, but it's required to
#             # make sure that attn_weights keeps its gradient.
#             # In order to do so, attn_weights have to be reshaped
#             # twice and have to be reused in the following
#             attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#             attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
#         else:
#             attn_weights_reshaped = None
#
#         attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)
#
#         attn_output = ops.bmm(attn_probs, value_states)
#
#         if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )
#
#         attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
#         attn_output = attn_output.transpose(1, 2)
#
#         # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
#         # partitioned aross GPUs when using tensor-parallelism.
#         attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
#
#         attn_output = self.out_proj(attn_output)
#
#         return attn_output, attn_weights_reshaped, past_key_value
#
#
# class SpeechT5FeedForward(nn.Cell):
#     def __init__(self, config, intermediate_size):
#         super().__init__()
#         self.intermediate_dropout = nn.Dropout(config.activation_dropout)
#
#         self.intermediate_dense = nn.Dense(config.hidden_size, intermediate_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act
#
#         self.output_dense = nn.Dense(intermediate_size, config.hidden_size)
#         self.output_dropout = nn.Dropout(config.hidden_dropout)
#
#     def construct(self, hidden_states):
#         hidden_states = self.intermediate_dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         hidden_states = self.intermediate_dropout(hidden_states)
#
#         hidden_states = self.output_dense(hidden_states)
#         hidden_states = self.output_dropout(hidden_states)
#         return hidden_states
#
#
# class SpeechT5EncoderLayer(nn.Cell):
#     def __init__(self, config: SpeechT5Config):
#         super().__init__()
#         self.attention = SpeechT5Attention(
#             embed_dim=config.hidden_size,
#             num_heads=config.encoder_attention_heads,
#             dropout=config.attention_dropout,
#             is_decoder=False,
#         )
#         self.dropout = nn.Dropout(config.hidden_dropout)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.feed_forward = SpeechT5FeedForward(config, config.encoder_ffn_dim)
#         self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#
#     def construct(
#         self,
#         hidden_states: ms.Tensor,
#         attention_mask: Optional[ms.Tensor] = None,
#         layer_head_mask: Optional[ms.Tensor] = None,
#         position_bias: Optional[ms.Tensor] = None,
#         output_attentions: bool = False,
#     ):
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`):
#                 input to the layer of shape `(batch, seq_len, hidden_size)`
#             attention_mask (`torch.FloatTensor`):
#                 attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
#                 large negative values.
#             layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
#                 `(config.encoder_attention_heads,)`.
#             position_bias (`torch.FloatTensor`):
#                 relative position embeddings of size `(seq_len, seq_len, hidden_size // encoder_attention_heads)`
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#         """
#         residual = hidden_states
#         hidden_states, attn_weights, _ = self.attention(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             layer_head_mask=layer_head_mask,
#             position_bias=position_bias,
#             output_attentions=output_attentions,
#         )
#
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = residual + hidden_states
#
#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = hidden_states + self.feed_forward(hidden_states)
#         hidden_states = self.final_layer_norm(hidden_states)
#
#         outputs = (hidden_states,)
#
#         if output_attentions:
#             outputs += (attn_weights,)
#
#         return outputs
#
#
# class SpeechT5DecoderLayer(nn.Cell):
#     def __init__(self, config: SpeechT5Config):
#         super().__init__()
#         self.self_attn = SpeechT5Attention(
#             embed_dim=config.hidden_size,
#             num_heads=config.decoder_attention_heads,
#             dropout=config.attention_dropout,
#             is_decoder=True,
#         )
#         self.dropout = nn.Dropout(config.hidden_dropout)
#         self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
#
#         self.encoder_attn = SpeechT5Attention(
#             config.hidden_size,
#             config.decoder_attention_heads,
#             dropout=config.attention_dropout,
#             is_decoder=True,
#         )
#         self.encoder_attn_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
#
#         self.feed_forward = SpeechT5FeedForward(config, config.decoder_ffn_dim)
#         self.final_layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
#
#     def construct(
#         self,
#         hidden_states: ms.Tensor,
#         attention_mask: Optional[ms.Tensor] = None,
#         encoder_hidden_states: Optional[ms.Tensor] = None,
#         encoder_attention_mask: Optional[ms.Tensor] = None,
#         layer_head_mask: Optional[ms.Tensor] = None,
#         cross_attn_layer_head_mask: Optional[ms.Tensor] = None,
#         past_key_value: Optional[Tuple[ms.Tensor]] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = True,
#     ):
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, hidden_size)`
#             attention_mask (`torch.FloatTensor`): attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#             encoder_hidden_states (`torch.FloatTensor`):
#                 cross attention input to the layer of shape `(batch, seq_len, hidden_size)`
#             encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#             layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
#                 `(encoder_attention_heads,)`.
#             cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
#                 size `(decoder_attention_heads,)`.
#             past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#         """
#         residual = hidden_states
#
#         # Self Attention
#         # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
#         self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
#         # add present self-attn cache to positions 1,2 of present_key_value tuple
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             past_key_value=self_attn_past_key_value,
#             attention_mask=attention_mask,
#             layer_head_mask=layer_head_mask,
#             output_attentions=output_attentions,
#         )
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = residual + hidden_states
#         hidden_states = self.self_attn_layer_norm(hidden_states)
#
#         # Cross-Attention Block
#         cross_attn_present_key_value = None
#         cross_attn_weights = None
#         if encoder_hidden_states is not None:
#             residual = hidden_states
#
#             # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
#             cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
#             hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
#                 hidden_states=hidden_states,
#                 key_value_states=encoder_hidden_states,
#                 attention_mask=encoder_attention_mask,
#                 layer_head_mask=cross_attn_layer_head_mask,
#                 past_key_value=cross_attn_past_key_value,
#                 output_attentions=output_attentions,
#             )
#             hidden_states = self.dropout(hidden_states)
#             hidden_states = residual + hidden_states
#             hidden_states = self.encoder_attn_layer_norm(hidden_states)
#
#             # add cross-attn to positions 3,4 of present_key_value tuple
#             present_key_value = present_key_value + cross_attn_present_key_value
#
#         # Fully Connected
#         hidden_states = hidden_states + self.feed_forward(hidden_states)
#         hidden_states = self.final_layer_norm(hidden_states)
#
#         outputs = (hidden_states,)
#
#         if output_attentions:
#             outputs += (self_attn_weights, cross_attn_weights)
#
#         if use_cache:
#             outputs += (present_key_value,)
#
#         return outputs
#
#
# class SpeechT5PreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """
#
#     config_class = SpeechT5Config
#     base_model_prefix = "speecht5"
#     main_input_name = "input_values"
#     supports_gradient_checkpointing = True
#
#     def _init_weights(self, module):
#         """Initialize the weights"""
#         if isinstance(module, SpeechT5PositionalConvEmbedding):
#             ops.normal(
#                 module.conv.weight,
#                 mean=0,
#                 std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
#             )
#             nn.init.constant_(module.conv.bias, 0)
#         elif isinstance(module, SpeechT5FeatureProjection):
#             k = math.sqrt(1 / module.projection.in_features)
#             ops.uniform(module.projection.weight, a=-k, b=k)
#             ops.uniform(module.projection.bias, a=-k, b=k)
#         elif isinstance(module, nn.Dense):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         elif isinstance(module, nn.Conv1d):
#             nn.init.kaiming_normal_(module.weight)
#             if module.bias is not None:
#                 k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
#                 nn.init.uniform_(module.bias, a=-k, b=k)
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()






class HifiGanResidualBlock(nn.Cell):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.CellList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.CellList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        for layer in self.convs1:
            weight_norm(layer)
        for layer in self.convs2:
            weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = ops.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = ops.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states



class SpeechT5HifiGan(PreTrainedModel):
    config_class = SpeechT5HifiGanConfig
    main_input_name = "spectrogram"

    def __init__(self, config: SpeechT5HifiGanConfig):
        super().__init__(config)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.CellList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.Conv1dTranspose(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.CellList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        self.register_buffer("mean", ops.zeros(config.model_in_dim))
        self.register_buffer("scale", ops.ones(config.model_in_dim))

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Dense, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def apply_weight_norm(self):
        weight_norm(self.conv_pre)
        for layer in self.upsampler:
            weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        weight_norm(self.conv_post)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for layer in self.upsampler:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            remove_weight_norm(layer)
        remove_weight_norm(self.conv_post)

    def construct(self, spectrogram: ms.Tensor) -> ms.Tensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        if self.config.normalize_before:
            spectrogram = (spectrogram - self.mean) / self.scale

        is_batched = spectrogram.dim() == 3
        if not is_batched:
            spectrogram = spectrogram.unsqueeze(0)

        hidden_states = spectrogram.transpose(2, 1)

        hidden_states = self.conv_pre(hidden_states)
        for i in range(self.num_upsamples):
            hidden_states = ops.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = ops.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        hidden_states = ops.tanh(hidden_states)

        if not is_batched:
            # remove batch dim and collapse tensor to 1-d audio waveform
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # remove seq-len dim since this collapses to 1
            waveform = hidden_states.squeeze(1)

        return waveform
