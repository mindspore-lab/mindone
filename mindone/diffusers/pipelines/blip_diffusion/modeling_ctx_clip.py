# Copyright 2024 Salesforce.com, inc.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import numpy as np
from transformers.models.clip.configuration_clip import CLIPTextConfig

import mindspore as ms
from mindspore import mint, nn

from mindone.transformers import CLIPPreTrainedModel
from mindone.transformers.modeling_outputs import BaseModelOutputWithPooling
from mindone.transformers.models.clip.modeling_clip import CLIPEncoder

_MIN_FP16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
_MIN_FP32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
_MIN_FP64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
_MIN_BF16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)


def dtype_to_min(dtype):
    if dtype == ms.float16:
        return _MIN_FP16
    if dtype == ms.float32:
        return _MIN_FP32
    if dtype == ms.float64:
        return _MIN_FP64
    if dtype == ms.bfloat16:
        return _MIN_BF16
    else:
        raise ValueError(f"Only support get minimum value of (float16, ), but got {dtype}")


def _expand_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].broadcast_to((bsz, 1, tgt_len, src_len)).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), dtype_to_min(dtype))


# This is a modified version of the CLIPTextModel from transformers.models.clip.modeling_clip
# Which allows for an extra input of "context embeddings", which are the query embeddings used in Qformer
# They pass through the clip model, along with the text embeddings, and interact with them using self attention
class ContextCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = ContextCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        ctx_embeddings: ms.Tensor = None,
        ctx_begin_pos: list = None,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return self.text_model(
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ContextCLIPTextTransformer(nn.Cell):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_return_dict = config.use_return_dict
        embed_dim = config.hidden_size
        self.embeddings = ContextCLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = mint.nn.LayerNorm((embed_dim,))

    def construct(
        self,
        ctx_embeddings: ms.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
        )

        bsz, seq_len = input_shape
        if ctx_embeddings is not None:
            seq_len += ctx_embeddings.shape[1]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        pooled_output = last_hidden_state[
            mint.arange(last_hidden_state.shape[0]),
            input_ids.argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = mint.zeros((bsz, seq_len, seq_len), dtype=dtype)
        mask = mask.fill(dtype_to_min(dtype))
        mask = mask.triu(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class ContextCLIPTextEmbeddings(nn.Cell):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = mint.nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = mint.nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = mint.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def construct(
        self,
        ctx_embeddings: ms.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        if ctx_embeddings is None:
            ctx_len = 0
        else:
            ctx_len = ctx_embeddings.shape[1]

        seq_length = (input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]) + ctx_len

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

            # for each input embeddings, add the ctx embeddings at the correct position
            input_embeds_ctx = []
            bsz = inputs_embeds.shape[0]

            if ctx_embeddings is not None:
                for i in range(bsz):
                    cbp = ctx_begin_pos[i]

                    prefix = inputs_embeds[i, :cbp]
                    # remove the special token embedding
                    suffix = inputs_embeds[i, cbp:]

                    input_embeds_ctx.append(mint.cat([prefix, ctx_embeddings[i], suffix], dim=0))

                inputs_embeds = mint.stack(input_embeds_ctx, dim=0)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
