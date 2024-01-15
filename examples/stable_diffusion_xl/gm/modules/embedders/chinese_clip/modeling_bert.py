# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# MindNLP BertModel compatible with HuggingFace transformers

import logging
import os
from collections import OrderedDict

from gm.modules.embedders.chinese_clip.configuration_bert import BertConfig

import mindspore.common.dtype as mstype
from mindspore import Tensor, nn, ops
from mindspore.amp import auto_mixed_precision
from mindspore.common.initializer import Normal, TruncatedNormal, initializer

__all__ = ["BertModel"]


class ClassInstantier(OrderedDict):
    r"""
    Class Instantier
    """

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    """
    Excitation equation matrix
    """
    "relu": nn.ReLU,
    "gelu": (nn.GELU, {"approximate": False}),
    "gelu_new": nn.GELU,
    "gelu_approximate": nn.GELU,
    "swish": nn.SiLU,
    "gelu_10": nn.GELU,
    "gelu_fast": nn.FastGelu,
    "gelu_python": nn.GELU,
    "linear": nn.ReLU,
    "mish": nn.Mish,
    "quick_gelu": nn.FastGelu,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


def torch_to_mindspore(pth_file, **kwargs):
    """convert torch checkpoint to mindspore"""
    _ = kwargs.get("prefix", "")

    try:
        import torch
    except Exception as exc:
        raise ImportError(
            "'import torch' failed, please install torch by "
            "`pip install torch` or instructions from 'https://pytorch.org'"
        ) from exc

    from mindspore.train.serialization import save_checkpoint

    logging.info("Starting checkpoint conversion.")
    ms_ckpt = []
    state_dict = torch.load(pth_file, map_location=torch.device("cpu"))

    for key, value in state_dict.items():
        if "LayerNorm" in key:
            if ".weight" in key:
                key = key.replace(".weight", ".gamma")
            if ".bias" in key:
                key = key.replace(".bias", ".beta")
        if "embeddings" in key:
            key = key.replace("weight", "embedding_table")
        if "self" in key:
            key = key.replace("self", "self_attn")
        ms_ckpt.append({"name": key, "data": Tensor(value.numpy())})

    ms_ckpt_path = pth_file.replace("pytorch_model.bin", "mindspore.ckpt")
    if not os.path.exists(ms_ckpt_path):
        try:
            save_checkpoint(ms_ckpt, ms_ckpt_path)
        except Exception as exc:
            raise RuntimeError(f"Save checkpoint to {ms_ckpt_path} failed, " f"please checkout the path.") from exc

    return ms_ckpt_path


class LayerNormFp32(nn.LayerNorm):
    """Subclass mindspore's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super(LayerNormFp32, self).__init__(*args, **kwargs)
        auto_mixed_precision(self, amp_level="O0")  # fp32

    def construct(self, x: Tensor):
        orig_type = x.dtype
        y, _, _ = self.layer_norm(
            x.astype(mstype.float32), self.gamma.astype(mstype.float32), self.beta.astype(mstype.float32)
        )
        return y.astype(orig_type)


class BertEmbeddings(nn.Cell):
    """
    Embeddings for BERT, include word, position and token_type
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_table=TruncatedNormal(config.initializer_range),
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embedding_table=TruncatedNormal(config.initializer_range),
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embedding_table=TruncatedNormal(config.initializer_range),
        )
        self.LayerNorm = LayerNormFp32((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Cell):
    """
    Self attention layer for BERT.
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(
            config.hidden_size,
            self.all_head_size,
            weight_init=TruncatedNormal(config.initializer_range),
        )
        self.key = nn.Dense(
            config.hidden_size,
            self.all_head_size,
            weight_init=TruncatedNormal(config.initializer_range),
        )
        self.value = nn.Dense(
            config.hidden_size,
            self.all_head_size,
            weight_init=TruncatedNormal(config.initializer_range),
        )

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

        self.softmax = nn.Softmax(-1)
        self.matmul = ops.BatchMatMul()

    def transpose_for_scores(self, input_x):
        r"""
        transpose for scores
        """
        new_x_shape = input_x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        input_x = input_x.view(*new_x_shape)
        return input_x.transpose(0, 2, 1, 3)

    def construct(self, hidden_states):
        # batch_size = hidden_states.shape[0]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_states = self.transpose_for_scores(mixed_query_layer)
        key_states = self.transpose_for_scores(mixed_key_layer)
        value_states = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" snd "key" to get the raw attention scores.
        attention_scores = self.matmul(query_states, key_states.swapaxes(-1, -2))
        attention_scores = attention_scores / ops.sqrt(Tensor(self.attention_head_size, mstype.float32))

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(0, 2, 1, 3)
        # new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.all_head_size)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Cell):
    r"""
    Bert Self Output
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
            weight_init=TruncatedNormal(config.initializer_range),
        )
        self.LayerNorm = LayerNormFp32((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Cell):
    r"""
    Bert Attention
    """

    def __init__(self, config):
        super().__init__()
        self.self_attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def construct(self, hidden_states):
        self_outputs = self.self_attn(hidden_states)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Cell):
    r"""
    Bert Intermediate
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.intermediate_size,
            weight_init=TruncatedNormal(config.initializer_range),
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Cell):
    r"""
    Bert Output
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.intermediate_size,
            config.hidden_size,
            weight_init=TruncatedNormal(config.initializer_range),
        )
        self.LayerNorm = LayerNormFp32((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutputForFlashAttention(nn.Cell):  # remove linear layer
    def __init__(self, config):
        super(BertSelfOutputForFlashAttention, self).__init__()
        self.LayerNorm = LayerNormFp32(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Cell):
    r"""
    Bert Layer
    """

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def construct(self, hidden_states):
        attention_outputs = self.attention(hidden_states)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertEncoder(nn.Cell):
    r"""
    Bert Encoder
    """

    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.CellList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, hidden_states):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions += (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs += (all_hidden_states,)
        if self.output_attentions:
            outputs += (all_attentions,)
        return outputs


class BertPooler(nn.Cell):
    r"""
    Bert Pooler
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(
            config.hidden_size,
            config.hidden_size,
            activation="tanh",
            weight_init=TruncatedNormal(config.initializer_range),
        )

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding.
        # to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


class BertPreTrainedModel(nn.Cell):
    """BertPretrainedModel"""

    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(
                initializer(
                    Normal(self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.has_bias:
                cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            embedding_table = initializer(
                Normal(self.config.initializer_range),
                cell.embedding_table.shape,
                cell.embedding_table.dtype,
            )
            if cell.padding_idx is not None:
                embedding_table[cell.padding_idx] = 0
            cell.embedding_table.set_data(embedding_table)
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer("ones", cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer("zeros", cell.beta.shape, cell.beta.dtype))


class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.pooler = BertPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def construct(self, input_ids):
        token_type_ids = ops.zeros_like(input_ids)
        position_ids = ops.broadcast_to(ops.arange(ops.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[0]
        # pooled_output = (
        #     self.pooler(sequence_output) if self.pooler is not None else None
        # )

        outputs = (
            sequence_output,
            None,  # pooled_output,
        ) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
