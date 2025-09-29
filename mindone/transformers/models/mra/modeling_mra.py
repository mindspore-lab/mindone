# coding=utf-8
# Copyright 2023 University of Wisconsin-Madison and The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
"""Mindspore MRA model."""

import math
from typing import Optional, Tuple, Union

from transformers.models.mra.configuration_mra import MraConfig
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

import mindspore
from mindspore.mint.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...mindspore_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "uw-madison/mra-base-512-4"
_CONFIG_FOR_DOC = "MraConfig"
_TOKENIZER_FOR_DOC = "AutoTokenizer"


mra_cuda_kernel = None


def sparse_mask(mask, indices, block_size=32):
    """
    Converts attention mask to a sparse mask for high resolution logits.
    """
    if len(mask.shape) != 2:
        raise ValueError("mask must be a 2-dimensional tensor.")

    if len(indices.shape) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")

    if mask.shape[0] != indices.shape[0]:
        raise ValueError("mask and indices must have the same size in the zero-th dimension.")

    batch_size, seq_len = mask.shape
    num_block = seq_len // block_size

    batch_idx = mindspore.mint.arange(
        indices.shape[0],
        dtype=mindspore.int64,
    )
    mask = mask.reshape(batch_size, num_block, block_size)
    mask = mask[batch_idx[:, None], (indices % num_block).long(), :]

    return mask


def transpose_indices(indices, dim_1_block, dim_2_block):
    return (
        (indices % dim_2_block) * dim_1_block + mindspore.mint.div(indices, dim_2_block, rounding_mode="floor")
    ).long()


class MraReduceSum:
    @staticmethod
    def operator_call(sparse_query, indices, query_num_block, key_num_block):
        batch_size, num_block, block_size, _ = sparse_query.shape

        if len(sparse_query.shape) != 4:
            raise ValueError("sparse_query must be a 4-dimensional tensor.")

        if len(indices.shape) != 2:
            raise ValueError("indices must be a 2-dimensional tensor.")

        _, _, block_size, _ = sparse_query.shape
        batch_size, num_block = indices.shape

        sparse_query = sparse_query.sum(dim=2).reshape(batch_size * num_block, block_size)

        batch_idx = mindspore.mint.arange(
            indices.shape[0],
            dtype=mindspore.int64,
        )
        global_idxes = (
            mindspore.mint.div(indices, key_num_block, rounding_mode="floor").long()
            + batch_idx[:, None] * query_num_block
        ).reshape(batch_size * num_block)
        temp = mindspore.mint.zeros(
            (batch_size * query_num_block, block_size),
            dtype=sparse_query.dtype,
        )
        output = temp.index_add(0, global_idxes, sparse_query).reshape(batch_size, query_num_block, block_size)

        output = output.reshape(batch_size, query_num_block * block_size)
        return output


def get_low_resolution_logit(query, key, block_size, mask=None, value=None):
    """
    Compute low resolution approximation.
    """
    batch_size, seq_len, head_dim = query.shape

    num_block_per_row = seq_len // block_size

    value_hat = None
    if mask is not None:
        token_count = mask.reshape(batch_size, num_block_per_row, block_size).sum(dim=-1)
        query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
            token_count[:, :, None] + 1e-6
        )
        key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
            token_count[:, :, None] + 1e-6
        )
        if value is not None:
            value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
                token_count[:, :, None] + 1e-6
            )
    else:
        token_count = block_size * mindspore.mint.ones(
            (batch_size, num_block_per_row),
            dtype=mindspore.float32,
        )
        query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        if value is not None:
            value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)

    low_resolution_logit = mindspore.mint.matmul(query_hat, key_hat.transpose(-1, -2)) / math.sqrt(head_dim)

    low_resolution_logit_row_max = low_resolution_logit.max(dim=-1, keepdims=True).values

    if mask is not None:
        low_resolution_logit = (
            low_resolution_logit - 1e4 * ((token_count[:, None, :] * token_count[:, :, None]) < 0.5).float()
        )

    return low_resolution_logit, token_count, low_resolution_logit_row_max, value_hat


def get_block_idxes(
    low_resolution_logit, num_blocks, approx_mode, initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks
):
    """
    Compute the indices of the subset of components to be used in the approximation.
    """
    batch_size, total_blocks_per_row, _ = low_resolution_logit.shape

    if initial_prior_diagonal_n_blocks > 0:
        offset = initial_prior_diagonal_n_blocks // 2
        temp_mask = mindspore.mint.ones((total_blocks_per_row, total_blocks_per_row))
        diagonal_mask = mindspore.mint.tril(mindspore.mint.triu(temp_mask, diagonal=-offset), diagonal=offset)
        low_resolution_logit = low_resolution_logit + diagonal_mask[None, :, :] * 5e3

    if initial_prior_first_n_blocks > 0:
        low_resolution_logit[:, :initial_prior_first_n_blocks, :] = (
            low_resolution_logit[:, :initial_prior_first_n_blocks, :] + 5e3
        )
        low_resolution_logit[:, :, :initial_prior_first_n_blocks] = (
            low_resolution_logit[:, :, :initial_prior_first_n_blocks] + 5e3
        )

    top_k_vals = mindspore.mint.topk(
        low_resolution_logit.reshape(batch_size, -1), num_blocks, dim=-1, largest=True, sorted=False
    )
    indices = top_k_vals.indices

    if approx_mode == "full":
        threshold = top_k_vals.values.min(dim=-1).values
        high_resolution_mask = (low_resolution_logit >= threshold[:, None, None]).float()
    elif approx_mode == "sparse":
        high_resolution_mask = None
    else:
        raise ValueError(f"{approx_mode} is not a valid approx_model value.")

    return indices, high_resolution_mask


def mra2_attention(
    query,
    key,
    value,
    mask,
    num_blocks,
    approx_mode,
    block_size=32,
    initial_prior_first_n_blocks=0,
    initial_prior_diagonal_n_blocks=0,
):
    """
    Use Mra to approximate self-attention.
    """
    if mra_cuda_kernel is None:
        return mindspore.mint.zeros_like(query)


class MraEmbeddings(mindspore.nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = mindspore.mint.nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = mindspore.mint.nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = mindspore.mint.nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = mindspore.mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mindspore.mint.nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", mindspore.mint.arange(config.max_position_embeddings).expand((1, -1)) + 2)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "token_type_ids",
            mindspore.mint.zeros(
                self.position_ids.shape,
                dtype=mindspore.int64,
            ),
            persistent=False,
        )

    def construct(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = mindspore.mint.zeros(
                    input_shape,
                    dtype=mindspore.int64,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MraSelfAttention(mindspore.nn.Cell):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = mindspore.mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = mindspore.mint.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = mindspore.mint.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = mindspore.mint.nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = (
            position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        )

        self.num_block = (config.max_position_embeddings // 32) * config.block_per_row
        self.num_block = min(self.num_block, int((config.max_position_embeddings // 32) ** 2))

        self.approx_mode = config.approx_mode
        self.initial_prior_first_n_blocks = config.initial_prior_first_n_blocks
        self.initial_prior_diagonal_n_blocks = config.initial_prior_diagonal_n_blocks

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        batch_size, num_heads, seq_len, head_dim = query_layer.shape

        # revert changes made by get_extended_attention_mask
        attention_mask = 1.0 + attention_mask / 10000.0
        attention_mask = attention_mask.squeeze().repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len).int()

        # The CUDA kernels are most efficient with inputs whose size is a multiple of a GPU's warp size (32). Inputs
        # smaller than this are padded with zeros.
        gpu_warp_size = 32

        if head_dim < gpu_warp_size:
            pad_size = batch_size, num_heads, seq_len, gpu_warp_size - head_dim

            query_layer = mindspore.mint.cat(
                [
                    query_layer,
                    mindspore.mint.zeros(
                        pad_size,
                    ),
                ],
                dim=-1,
            )
            key_layer = mindspore.mint.cat(
                [
                    key_layer,
                    mindspore.mint.zeros(
                        pad_size,
                    ),
                ],
                dim=-1,
            )
            value_layer = mindspore.mint.cat(
                [
                    value_layer,
                    mindspore.mint.zeros(
                        pad_size,
                    ),
                ],
                dim=-1,
            )

        context_layer = mra2_attention(
            query_layer.float(),
            key_layer.float(),
            value_layer.float(),
            attention_mask.float(),
            self.num_block,
            approx_mode=self.approx_mode,
            initial_prior_first_n_blocks=self.initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks=self.initial_prior_diagonal_n_blocks,
        )

        if head_dim < gpu_warp_size:
            context_layer = context_layer[:, :, :, :head_dim]

        context_layer = context_layer.reshape(batch_size, num_heads, seq_len, head_dim)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class MraSelfOutput(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mindspore.mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = mindspore.mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mindspore.mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MraAttention(mindspore.nn.Cell):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = MraSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = MraSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class MraIntermediate(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mindspore.mint.nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class MraOutput(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mindspore.mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = mindspore.mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = mindspore.mint.nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MraLayer(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MraAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = MraIntermediate(config)
        self.output = MraOutput(config)

    def construct(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MraEncoder(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = mindspore.nn.CellList([MraLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform
class MraPredictionHeadTransform(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = mindspore.mint.nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = mindspore.mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Mra
class MraLMPredictionHead(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = MraPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = mindspore.mint.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = mindspore.Parameter(mindspore.mint.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Mra
class MraOnlyMLMHead(mindspore.nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = MraLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.yoso.modeling_yoso.YosoPreTrainedModel with Yoso->Mra,yoso->mra
class MraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MraConfig
    base_model_prefix = "mra"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, mindspore.mint.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, mindspore.mint.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, mindspore.mint.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MRA_START_DOCSTRING = r"""
    This model is a Mindspore [mindspore.nn.Cell] sub-class. Use
    it as a regular Mindspore Module and refer to the Mindspore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`mindspore.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`mindspore.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`mindspore.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`mindspore.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MRA Model transformer outputting raw hidden-states without any specific head on top.",
    MRA_START_DOCSTRING,
)
class MraModel(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = MraEmbeddings(config)
        self.encoder = MraEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = mindspore.mint.ones(
                ((batch_size, seq_length)),
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = mindspore.mint.zeros(
                    input_shape,
                    dtype=mindspore.int64,
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings("""MRA Model with a `language modeling` head on top.""", MRA_START_DOCSTRING)
class MraForMaskedLM(MraPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.mra = MraModel(config)
        self.cls = MraOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.yoso.modeling_yoso.YosoClassificationHead with Yoso->Mra
class MraClassificationHead(mindspore.nn.Cell):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = mindspore.mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = mindspore.mint.nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = mindspore.mint.nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def construct(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """MRA Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    MRA_START_DOCSTRING,
)
class MraForSequenceClassification(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mra = MraModel(config)
        self.classifier = MraClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == mindspore.int64 or labels.dtype == mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """MRA Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    MRA_START_DOCSTRING,
)
class MraForMultipleChoice(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.mra = MraModel(config)
        self.pre_classifier = mindspore.mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = mindspore.mint.nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = mindspore.mint.nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """MRA Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.""",
    MRA_START_DOCSTRING,
)
class MraForTokenClassification(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mra = MraModel(config)
        self.dropout = mindspore.mint.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = mindspore.mint.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = mindspore.mint.where(
                    active_loss, labels.view(-1), mindspore.Tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """MRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    MRA_START_DOCSTRING,
)
class MraForQuestionAnswering(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.mra = MraModel(config)
        self.qa_outputs = mindspore.mint.nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "MraForMaskedLM",
    "MraForMultipleChoice",
    "MraForQuestionAnswering",
    "MraForSequenceClassification",
    "MraForTokenClassification",
    "MraLayer",
    "MraModel",
    "MraPreTrainedModel",
]
