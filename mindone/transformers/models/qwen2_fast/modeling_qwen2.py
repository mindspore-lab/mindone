# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Mindspore Qwen2 fast infer model."""

import time
import math
import numpy as np
from typing import List, Optional, Tuple, Union

from transformers import Qwen2Config, logging

import mindspore
from mindspore import Parameter, mint, nn, ops, Tensor
from mindspore.ops import operations as P
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mindspore.common import lazy_inline
from mindspore.common.initializer import initializer

from mindone.transformers.cache_utils import get_seq_length, get_max_length, update
from mindone.transformers.modeling_attn_mask_utils import AttentionMaskConverter, dtype_to_min, _MIN_FP16
from mindone.transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from mindone.transformers.modeling_utils import MSPreTrainedModel
from mindone.transformers.mindspore_adapter.select_operator import get_multinomial_op

from .page_attention.infer_attention import InferAttention
from .page_attention.block_tables import BlockTables
from .fast_modules.linear import FastLinear
from .fast_modules.activation import FastSiLU
from .fast_modules.freqs import FreqsMgr
from .fast_modules.mask import LowerTriangularMaskWithDynamic
from .fast_modules.embedding import FastEmbedding
from .utils.version_control import check_rmsnorm_big_kernel_valid
from .utils.utils import is_pynative

# for sampling
from mindone.transformers.generation.logits_process import LogitsProcessorList
from mindone.transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateNonBeamOutput


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"


class FastQwen2RMSNorm(nn.Cell):

    def __init__(self, dim, eps=1e-6, compute_type=mindspore.float32, fused_kernel=True):
        super(FastQwen2RMSNorm, self).__init__()
        self.eps = eps
        self.compute_type = compute_type
        self.weight = Parameter(initializer('ones', (dim,), dtype=self.compute_type), parallel_optimizer=False)

        if fused_kernel and check_rmsnorm_big_kernel_valid():
            self.norm = P.RmsNorm(eps)
            self.rms_norm = self._rms_norm
            self.self_define = False
            self.cast = P.Cast()
            self.rcast = P.Cast()
        else:
            self.cast = P.Cast()
            self.mul = P.Mul()
            self.mul2 = P.Mul()
            self.square = P.Square()
            self.mean = P.ReduceMean(keep_dims=True)
            self.add = P.Add()
            self.rsqrt = P.Rsqrt()
            # self.rms_norm = self._self_norm
            self.rms_norm = self._ops_norm
            self.self_define = True

    def _ops_norm(self, x):
        original_type = x.dtype
        output, _ = ops.rms_norm(
            x.to(self.compute_type), self.weight.to(self.compute_type), epsilon=self.eps
        )
        output = output.to(original_type)
        return output

    def _self_norm(self, x):
        original_type = x.dtype
        h = self.cast(x, self.compute_type)
        norm_factor = self.square(h)
        norm_factor = self.mean(norm_factor, -1)
        norm_factor = self.add(norm_factor, self.eps)
        norm_factor = self.rsqrt(norm_factor)
        output = self.mul(h, norm_factor)
        output = self.mul2(self.cast(output, original_type), self.cast(self.weight, original_type))
        return output

    def _rms_norm(self, x):
        original_type = x.dtype
        output = self.norm(self.cast(x, self.compute_type), self.weight)[0]
        return self.rcast(output, original_type)

    def construct(self, x):
        """Forward of RMSNorm."""
        return self.rms_norm(x)


class FastQwen2MLP(nn.Cell):
    def __init__(
        self,
        config,
        compute_dtype=mindspore.float16,
        param_init_type=mindspore.float32,
        ffn_concat=False,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.compute_dtype = compute_dtype
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.reshape = P.Reshape()

        if not ffn_concat:
            self.gate_proj = FastLinear(self.hidden_size, self.intermediate_size, has_bias=False, compute_dtype=compute_dtype, param_init_type=param_init_type)
            self.up_proj = FastLinear(self.hidden_size, self.intermediate_size, has_bias=False, compute_dtype=compute_dtype, param_init_type=param_init_type)
            self.down_proj = FastLinear(self.intermediate_size, self.hidden_size, has_bias=False, compute_dtype=compute_dtype, param_init_type=param_init_type)
            self.act_fn = FastSiLU()
        else:
            raise NotImplementedError

    def construct(self, hidden_state):
        # TODO: support ffn_concat
        hidden_state = self.cast(hidden_state, self.compute_dtype)
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class FastQwen2Attention(nn.Cell):
    def __init__(
            self,
            config: Qwen2Config,
            layer_idx: Optional[int] = None,
            *,
            max_seq_length=32768,
            is_dynamic=True,
            use_cache=True,
            use_flash_attention=True,
            qkv_concat=False,
            block_size: Optional[int] = None,
            num_blocks: Optional[int] = None,
            compute_dtype=mindspore.bfloat16,
            param_init_type=mindspore.bfloat16,
        ):

        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # self.max_position_embeddings = config.max_position_embeddings
        # self.rope_theta = config.rope_theta
        # self.is_causal = True
        # self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.is_first_iteration = True
        self.compute_dtype = compute_dtype
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        if qkv_concat:
            raise NotImplementedError
        else:
            self.q_proj = FastLinear(self.hidden_size, self.num_heads * self.head_dim, has_bias=True, compute_dtype=compute_dtype, param_init_type=param_init_type)
            self.k_proj = FastLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True, compute_dtype=compute_dtype, param_init_type=param_init_type)
            self.v_proj = FastLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True, compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.o_proj = FastLinear(self.num_heads * self.head_dim, self.hidden_size, has_bias=False, compute_dtype=compute_dtype, param_init_type=param_init_type)

        if use_cache:
            self.infer_attention = InferAttention(self.num_heads,
                                                  self.head_dim,
                                                  self.num_key_value_heads,
                                                  seq_length=max_seq_length,
                                                  pa_n_head_split=self.num_heads,
                                                  pa_n_kv_head_split=self.num_key_value_heads,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  pre_tokens=2147483647,
                                                  next_tokens=0,
                                                  block_size=self.block_size,
                                                  num_blocks=self.num_blocks,
                                                  is_dynamic=is_dynamic,
                                                  use_flash_attention=use_flash_attention,
                                                  rotary_cos_format=2,
                                                  compute_dtype=compute_dtype
                                                  )
        else:
            raise NotImplementedError

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        freqs_cis: tuple[mindspore.Tensor, mindspore.Tensor] = None,
        mask=None,
        batch_valid_length=None,
        block_tables=None,
        slot_mapping=None,
        prefix_keys_values=None,
        q_seq_lens=None,
    ):
        ori_dtype = hidden_states.dtype

        # TODO: Add qkv-concat
        # if not qkv-concat
        query = self.cast(self.q_proj(hidden_states), self.compute_dtype)
        key = self.cast(self.k_proj(hidden_states), self.compute_dtype)
        value = self.cast(self.v_proj(hidden_states), self.compute_dtype)
        # key and value for current token(s)

        # TODO: Add attention compute w/o kv-cache
        # if use_cache
        context_layer = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                             freqs_cis, mask, prefix_keys_values=prefix_keys_values,
                                             q_seq_lens=q_seq_lens)
        output = self.o_proj(context_layer)
        output = self.cast(output, ori_dtype)
        return output


class FastQwen2DecoderLayer(nn.Cell):

    @lazy_inline
    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        *,
        max_seq_length: int,
        use_cache=True,
        is_dynamic=True,
        use_flash_attention=False,
        block_size: Optional[int] = None,
        num_blocks: Optional[int] = None,
        fused_kernel=True,
        qkv_concat=False,
        compute_dtype=mindspore.float16,
        layernorm_compute_dtype=mindspore.float32,
        param_init_type=mindspore.float32,
        residual_dtype=mindspore.float32,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        
        self.compute_dtype = compute_dtype
        self.is_first_iteration = True
        self.residual_dtype = residual_dtype
        self.residual_cast_flag = residual_dtype != compute_dtype
        if self.residual_cast_flag:
            logger.info(f"residual cast flag: {self.residual_cast_flag}, residual dtype: {residual_dtype}")
        self.add = P.Add()
        self.cast = P.Cast()

        self.self_attn = FastQwen2Attention(
            config,
            layer_idx,
            max_seq_length=max_seq_length,
            is_dynamic=is_dynamic,
            use_cache=use_cache,
            use_flash_attention=use_flash_attention,
            qkv_concat=qkv_concat,
            block_size=block_size,
            num_blocks=num_blocks,
            compute_dtype=compute_dtype,
            param_init_type=param_init_type
        )

        self.mlp = FastQwen2MLP(config, compute_dtype=compute_dtype, param_init_type=param_init_type, ffn_concat=qkv_concat)
        self.input_layernorm = FastQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, compute_type=layernorm_compute_dtype, fused_kernel=fused_kernel)
        self.post_attention_layernorm = FastQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, compute_type=layernorm_compute_dtype, fused_kernel=fused_kernel)

    def construct(
        self,
        x,
        freqs_cis,
        mask=None,
        batch_valid_length=None,
        block_tables=None,
        slot_mapping=None,
        prefix_keys_values=None,
        q_seq_lens=None,
    ):
        # [bs, seq/1, hidden_dim]
        input_x = self.input_layernorm(x)
        # [bs, seq/1, hidden_dim]
        h = self.self_attn(input_x, freqs_cis, mask, batch_valid_length, block_tables,
                           slot_mapping, prefix_keys_values, q_seq_lens)

        if self.residual_cast_flag:
            x = self.cast(x, self.residual_dtype)
            h = self.cast(h, self.residual_dtype)

        h = self.add(x, h)
        if self.residual_cast_flag:
            h = self.cast(h, self.compute_dtype)
        ffn_norm = self.post_attention_layernorm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.mlp(ffn_norm)
        if self.residual_cast_flag:
            h = self.cast(h, self.residual_dtype)
            ffn_out = self.cast(ffn_out, self.residual_dtype)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)
        if self.residual_cast_flag:
            out = self.cast(out, self.compute_dtype)
        return out


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Cell](https://pytorch.org/docs/stable/nn.html#torch.nn.Cell) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


class Qwen2PreTrainedModel(MSPreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = False  # FIXME

    def _init_weights(self, module):
        # std = self.config.initializer_range
        # if isinstance(module, nn.Dense):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        pass


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(mindspore.Tensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`mindspore.Tensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class FastInferQwen2Model(Qwen2PreTrainedModel):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        is_dynamic=True,
        use_flash_attention=True,
        batch_size=1,
        max_seq_length=32768,
        max_position_embedding=32768,
        block_size=32,
        num_blocks=1024,
        compute_dtype=mindspore.bfloat16,
        residual_dtype=None,
        layernorm_compute_type=mindspore.float32,
        param_init_type=mindspore.bfloat16,
        embedding_init_type=None,
        rotary_dtype=mindspore.bfloat16,
        scaling_factor=1.0,
        extend_method="None",
        qkv_concat=False,
        fused_rms_norm=True,
    ):
        super().__init__(config)
        self.vocab_size = config.vocab_size

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        use_cache = config.use_cache
        if not use_cache:
            raise NotImplementedError(f"FastInferQwen2Model only support infer with kv-cache, please set `use_cache=true` in config.")

        self.max_seq_length = max_seq_length
        self.use_cache = use_cache
        self.is_dynamic = is_dynamic
        self.compute_dtype = compute_dtype
        self.is_first_iteration = True
        self.use_flash_attention = use_flash_attention
        self.concat = P.Concat(-1)
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.num_blocks = num_blocks
        self.block_size = block_size
        residual_dtype = residual_dtype if residual_dtype is not None else compute_dtype
        embedding_init_type = embedding_init_type if embedding_init_type is not None else param_init_type
        self.residual_dtype = residual_dtype
        self.embedding_init_type = embedding_init_type

        self.freqs_mgr = FreqsMgr(head_dim=self.head_dim,
                                  seq_length=max_seq_length,
                                  max_position_embedding=max_position_embedding,
                                  rotary_dtype=rotary_dtype,
                                  theta=config.rope_theta,
                                  scaling_factor=scaling_factor,
                                  extend_method=extend_method,
                                  is_dynamic=is_dynamic)
        self.residual_cast_flag = self.residual_dtype != self.compute_dtype
        if self.residual_cast_flag:
            logger.info(f"residual in llama model cast flag: {self.residual_cast_flag}, "
                        f"residual dtype: {self.residual_dtype}")

        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=max_seq_length,
                                                          batch_size=batch_size,
                                                          compute_type=compute_dtype,
                                                          is_dynamic=is_dynamic,
                                                          pad_token_id=config.pad_token_id,
                                                          use_flash_attention=use_flash_attention,
                                                          use_attn_mask_compression=False,
                                                          use_past=self.use_cache,
                                                          chunk_prefill=False)

        self.embed_tokens = FastEmbedding(vocab_table_size=config.vocab_size,
                                          embedding_size=config.hidden_size,
                                          param_init_type=self.embedding_init_type,
                                          rmsnorm_compute_2d=False)


        self.layers = nn.CellList(
            [
                FastQwen2DecoderLayer(
                    config,
                    layer_idx,
                    max_seq_length=max_seq_length,
                    use_cache=use_cache,
                    is_dynamic=is_dynamic,
                    use_flash_attention=use_flash_attention,
                    block_size=block_size,
                    num_blocks=num_blocks,
                    fused_kernel=fused_rms_norm,
                    qkv_concat=qkv_concat,
                    compute_dtype=compute_dtype,
                    layernorm_compute_dtype=layernorm_compute_type,
                    param_init_type=param_init_type,
                    residual_dtype=residual_dtype,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = FastQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, compute_type=layernorm_compute_type, fused_kernel=fused_rms_norm)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def construct(
        self,
        input_ids: Tensor,
        input_embeds=None,
        batch_valid_length=None,
        block_tables=None,
        slot_mapping=None,
        prefix_keys_values=None,
        attention_mask=None,
        position_ids=None,
        q_seq_lens=None,
    ):
        # preprocess
        bs, seq_len = self.shape(input_ids)
        rmsnorm_compute_2d = False
        if attention_mask is not None:
            mask = attention_mask
            mask = self.cast(mask, mindspore.uint8)
            freqs_cis = self.freqs_mgr(seq_len, position_ids)
        else:
            mask = None
            
            # TODO: only support kv-cahce
            # if self.use_cache:
            if self.is_first_iteration:
                freqs_cis = self.freqs_mgr.prefill(bs, seq_len)
                if self.use_flash_attention:
                    mask = self.casual_mask.prefill()
                else:
                    mask = self.casual_mask(input_ids)
                if prefix_keys_values is not None:
                    if mask is None:
                        mask = self.casual_mask(input_ids)
                    prefix_length = prefix_keys_values[0].shape[2]
                    prefix_mask = Tensor(np.zeros((bs, 1, seq_len, prefix_length)), dtype=mask.dtype)
                    mask = self.concat((prefix_mask, mask))
            else:
                freqs_cis = self.freqs_mgr.increment(batch_valid_length)

        if input_embeds is not None:
            h = self.cast(input_embeds, self.compute_dtype)
        else:
            h = self.cast(self.embed_tokens(input_ids), self.compute_dtype)
        if not rmsnorm_compute_2d:
            h = self.reshape(h, (bs, seq_len, self.hidden_size))    # h: [bs, seq/1, hidden_dim]
        for i in range(self.num_layers):
            prefix_kv = prefix_keys_values[i] if prefix_keys_values is not None else None
            h = self.layers[i](
                h,
                freqs_cis,
                mask,
                batch_valid_length=batch_valid_length,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                prefix_keys_values=prefix_kv,
                q_seq_lens=q_seq_lens,
            )

        if rmsnorm_compute_2d:
            h = self.reshape(h, (bs * seq_len, -1))
        output = self.norm(h)
        return output


class FastInferQwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: Qwen2Config,
        *,
        is_dynamic=True,
        use_flash_attention=True,
        batch_size=1,
        max_seq_length=32768,
        max_position_embedding=32768,
        block_size=32,
        num_blocks=1024,
        compute_dtype=mindspore.bfloat16,
        residual_dtype=None,
        layernorm_compute_type=mindspore.float32,
        param_init_type=mindspore.bfloat16,
        embedding_init_type=None,
        rotary_dtype=mindspore.bfloat16,
        scaling_factor=1.0,
        extend_method="None",
        qkv_concat=False,
        fused_rms_norm=True,
    ):
        super().__init__(config)
        
        self.vocab_size = config.vocab_size
        self.use_cache = config.use_cache
        self.ignore_token_id = config.ignore_token_id if hasattr(config, "ignore_token_id") else -100
        self.pad_token_id = config.pad_token_id

        self.model = FastInferQwen2Model(
            config,
            is_dynamic=is_dynamic,
            use_flash_attention=use_flash_attention,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_position_embedding=max_position_embedding,
            block_size=block_size,
            num_blocks=num_blocks,
            compute_dtype=compute_dtype,
            residual_dtype=residual_dtype,
            layernorm_compute_type=layernorm_compute_type,
            param_init_type=param_init_type,
            embedding_init_type=embedding_init_type,
            rotary_dtype=rotary_dtype,
            scaling_factor=scaling_factor,
            extend_method=extend_method,
            qkv_concat=qkv_concat,
            fused_rms_norm=fused_rms_norm,
        )
        self.lm_head = FastLinear(config.hidden_size, config.vocab_size, has_bias=False,
                                  compute_dtype=compute_dtype, param_init_type=param_init_type)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.max_seq_length = max_seq_length
        self.is_dynamic = is_dynamic
        self.is_first_iteration = True
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.add = P.Add()
        self.ones = P.Ones()
        self.gather = P.Gather(1)
        self.prefill_gather_flatten = P.Gather()
        self.sub_batch_valid_len = P.Sub()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def enable_dynamic_shape(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mindspore.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mindspore.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mindspore.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mindspore.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=mindspore.float16)
            self.set_inputs(
                dynamic_input_ids,
                None,
                None,
                None,
                None,
                dynamic_batch_valid_length,
                dynamic_block_tables,
                dynamic_slot_mapping,
                dynamic_prefix_keys_values,
                None,
                None,
                None
            )
        elif self.use_cache:
            self.set_inputs(
                dynamic_input_ids,
                None,
                None,
                None,
                None,
                dynamic_batch_valid_length,
                dynamic_block_tables,
                dynamic_slot_mapping,
                None,
                None,
                None,
                None
            )
        else:
            raise NotImplementedError
        logger.info("Set dynamic input.")

    def construct(
        self,
        input_ids,
        labels=None,
        position_ids=None,
        attention_mask=None,
        input_embeds=None,
        batch_valid_length=None,
        block_tables=None,
        slot_mapping=None,
        prefix_keys_values=None,
        q_seq_lens=None,
        loss_mask=None,
        gather_index=None,
    ):
        r"""FastInferQwen2ForCausalLM forward."""
        has_loss_mask = loss_mask is not None

        bsz, seqlen = self.shape(input_ids)
        if self.use_cache:
            if not isinstance(batch_valid_length, Tensor):
                batch_valid_length = self.ones((bsz,), mindspore.int32)
        tokens = input_ids
        if batch_valid_length is not None:
            batch_valid_length = self.reshape(batch_valid_length, (-1,))

        output = self.model(
            tokens,
            input_embeds,
            batch_valid_length,
            block_tables,
            slot_mapping,
            prefix_keys_values,
            attention_mask,
            position_ids,
            q_seq_lens,
        )

        pre_gather = (not self.use_cache or self.is_first_iteration) and batch_valid_length is not None
        output = self.pre_gather_func(pre_gather, output, batch_valid_length, gather_index)
        logits = self.lm_head(output)
        input_mask = loss_mask if has_loss_mask \
            else self.cast(self.not_equal(tokens, self.pad_token_id), mindspore.float32)

        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bsz, seqlen), (1, 1))
        else:
            if labels.ndim > 1:
                if not has_loss_mask:
                    label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), mindspore.float32)
                    input_mask = self.mul(input_mask, label_mask)

        logits = self.cast(logits, mindspore.float32)
        logits = self.reshape(logits, (-1, logits.shape[-1]))
        return logits

    def pre_gather_func(self, pre_gather, output, batch_valid_length, gather_index=None):
        """Pre gather operation in infer mode."""
        if not pre_gather:
            return output
        if pre_gather:
            if self.is_dynamic:
                batch_valid_length = mint.cumsum(batch_valid_length, 0)
                output = self.prefill_gather_flatten(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
            else:
                output = self.gather(output, self.sub_batch_valid_len(batch_valid_length, 1), 1)
        return output

    def add_flags_custom(self, is_first_iteration):
        """Add customized attributes for specific cells in the model."""
        self.add_flags(is_first_iteration=is_first_iteration)
        self.model.add_flags(is_first_iteration=is_first_iteration)
        for layer in self.model.layers:
            layer.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.add_flags(is_first_iteration=is_first_iteration)
            layer.self_attn.infer_attention.paged_attention_mgr.add_flags(is_first_iteration=is_first_iteration)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        prepare inputs for generation.
        A model class needs to define a `prepare_inputs_for_generation` method
        in order to use `.generate()`

        """
        model_inputs = {"input_ids": Tensor.from_numpy(input_ids.astype(np.int32))}
        if self.model.is_dynamic:
            prefill = kwargs.get("prefill")
            if prefill and "origin_inputs" in kwargs:
                origin_inputs = kwargs["origin_inputs"]
                batch_valid_length = kwargs.get("valid_length_each_example")
                slot_mapping = kwargs.get("slot_mapping")
                model_inputs = self._prepare_inputs_for_prefill_flatten(origin_inputs,
                                                                        batch_valid_length,
                                                                        slot_mapping,
                                                                        model_inputs)
        return model_inputs

    def _sample(
        self,
        input_ids: mindspore.Tensor,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus,
        streamer,
        logits_warper,
        **model_kwargs,
    ):
        if isinstance(input_ids, mindspore.Tensor):
            input_ids = input_ids.asnumpy()

        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = ops.ones(batch_size, dtype=mindspore.int32)
        # model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        valid_length_each_example, input_ids_length = \
            self.get_valid_length_each_example(input_ids, generation_config.pad_token_id)
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length
        if generation_config.max_length > self.max_seq_length:
            logger.warning("max_length %s can not exceeds model seq_length %s, set max_length = seq_length.",
                           generation_config.max_length, self.max_seq_length)
            generation_config.max_length = self.max_seq_length
        if generation_config.max_new_tokens is not None:
            max_length_each_example = [valid_length + generation_config.max_new_tokens \
                for valid_length in valid_length_each_example]
        else:
            max_length_each_example = [generation_config.max_length] * len(valid_length_each_example)

        multinomial = get_multinomial_op()
        step = 0
        s_time = time.time()
        graph_compiled_time_buffer = []

        origin_inputs = input_ids
        input_ids = self._pad_inputs_using_max_length(
            origin_inputs=origin_inputs, pad_token_id=generation_config.pad_token_id, max_seq_length=self.max_seq_length
        )
        valid_length_each_example, _ = \
            self.get_valid_length_each_example(origin_inputs, generation_config.pad_token_id)
        batch_size = origin_inputs.shape[0]

        use_dynamic = self.model.is_dynamic #model_kwargs.pop("enable_dynamic_shape", True)
        if not use_dynamic:
            raise NotImplementedError

        use_cache = model_kwargs["use_cache"]
        if not use_cache:
            raise NotImplementedError

        # call mindspore.nn.Cell.set_inputs() function
        self.enable_dynamic_shape()
        self._set_block_mgr(batch_size, self.max_seq_length, self.model.num_blocks, self.model.block_size)

        prefill = True
        model_kwargs["origin_inputs"] = origin_inputs

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus):

            is_finished = (~unfinished_sequences).asnumpy().astype(np.bool_).tolist()
            block_tables = None
            slot_mapping = None
            if prefill:
                if (is_pynative() and use_dynamic):
                    max_input_length = len(origin_inputs[0])
                else:
                    max_input_length = self.max_seq_length
                block_tables, slot_mapping = self.block_mgr.assemble_pa_full_inputs(max_input_length,
                                                                                    valid_length_each_example,
                                                                                    is_finished)
            else:
                block_tables, slot_mapping = self.block_mgr.assemble_pa_inc_inputs(valid_length_each_example,
                                                                                   is_finished)

            #
            max_valid_length = max(valid_length_each_example)
            if not self.config.is_encoder_decoder and max_valid_length > self.max_seq_length:
                raise ValueError(
                    f"The input length:{max_valid_length} is longer than the seq_length:{self.max_seq_length}, "
                    "which is not allowed."
                )

            input_ids = np.reshape(input_ids, (-1, np.shape(input_ids)[-1]))

            # prepare model inputs
            current_index = valid_length_each_example - 1 + np.arange(input_ids.size, step=input_ids.shape[1])
            model_kwargs["current_index"] = current_index
            model_kwargs["prefill"] = prefill
            model_kwargs["valid_length_each_example"] = valid_length_each_example
            model_kwargs["block_tables"] = block_tables
            model_kwargs["slot_mapping"] = slot_mapping
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            real_input_ids = model_inputs["input_ids"]
            current_index = valid_length_each_example - 1 + np.arange(real_input_ids.numel(), step=real_input_ids.shape[1])
            if "batch_valid_length" not in model_inputs:
                model_inputs["batch_valid_length"] = Tensor.from_numpy(
                    np.array([valid_length_each_example], dtype=np.int32))
            if block_tables is not None and "block_tables" not in model_inputs:
                model_inputs["block_tables"] = Tensor.from_numpy(block_tables)
            if slot_mapping is not None and "slot_mapping" not in model_inputs:
                model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)

            if prefill:
                self.phase = "prefill"
                self.add_flags_custom(is_first_iteration=True)
            else:
                model_inputs = self.slice_incremental_inputs(model_inputs, current_index)

            # run forward
            s_time = time.time()
            outputs = self(
                **model_inputs
            )
            print(f"model infer time: {(time.time()-s_time)*1000:.2f} ms")

            if prefill:
                self.phase = "increment"
                self.add_flags_custom(is_first_iteration=False)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits.reshape(-1, logits.shape[-1])  # (bs*?, dim)
            
            next_token_logits = logits[:] # copy
            need_gather_logits = prefill if not self.config.is_encoder_decoder and use_cache else True
            if need_gather_logits and next_token_logits.shape[0] > len(current_index):
                next_token_logits = next_token_logits[Tensor(current_index, dtype=mindspore.int32)]  # (bs, dim)

            if use_cache:
                if prefill and "origin_inputs" in model_kwargs:
                    model_kwargs.pop("origin_inputs")
                prefill = False
            else:
                raise NotImplementedError

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            step_time = time.time() - s_time
            if step < 2:
                print(f"==> sampling, step: {step}, time cost: {step_time:.5f}s")
            else:
                graph_compiled_time_buffer.append(step_time)
                token_speed = len(graph_compiled_time_buffer) / sum(graph_compiled_time_buffer)
                print(
                    f"==> sampling, step: {step}, time cost: {step_time:.5f}s, running avg speed: {token_speed:.5f}token/s"
                )
            s_time = time.time()
            step += 1

            if not isinstance(outputs, CausalLMOutputWithPast):
                outputs = CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                )

            # zhy_test: numpy->tensor for logits_processor
            # pre-process distribution
            next_token_scores = logits_processor(mindspore.Tensor(input_ids), next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(mindspore.Tensor(input_ids), next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = ops.softmax(next_token_scores, axis=-1, dtype=mindspore.float32).to(next_token_scores.dtype)
                next_tokens = multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = ops.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            next_tokens = next_tokens.to(mindspore.int32)
            next_tokens = next_tokens.asnumpy()

            for i in range(batch_size):
                if is_finished[i]:
                    continue

                input_ids[i, valid_length_each_example[i]] = next_tokens[i]

                if self.config.is_encoder_decoder:
                    raise NotImplementedError

                # Stop judgment
                if next_tokens[i] in generation_config.eos_token_id \
                        or valid_length_each_example[i] + 1 == generation_config.max_length \
                        or valid_length_each_example[i] + 1 == max_length_each_example[i]:
                    is_finished[i] = True
                    unfinished_sequences = unfinished_sequences & ~mindspore.Tensor(np.array(is_finished), mindspore.bool_)
                else:
                    valid_length_each_example[i] += 1

            # update generated ids, model inputs, and length for next step
            if streamer is not None:
                streamer.put(next_tokens)

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs,
            #     model_kwargs,
            #     is_encoder_decoder=self.config.is_encoder_decoder,
            # )

            # zhy_test, input_ids clip to max_valid_length
            unfinished_sequences = unfinished_sequences & ~mindspore.Tensor(stopping_criteria(input_ids[:, :int(valid_length_each_example.max())], scores), mindspore.bool_)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _pad_inputs_using_max_length(self, origin_inputs, pad_token_id=0, max_seq_length=None):
        """pad the input_ids to the max_length"""
        pad_length = max_seq_length - origin_inputs.shape[-1]
        if pad_length < 0:
            raise ValueError(
                f"origin_inputs size is {origin_inputs.shape}, you should"
                f"increase the seq_length of the model {max_seq_length}."
            )
        # Pad original inputs to model_origin_max_length
        input_ids = np.pad(
            origin_inputs,
            ((0, 0), (0, pad_length)),
            "constant",
            constant_values=(0, pad_token_id),
        )
        return input_ids

    def slice_incremental_inputs(self, model_inputs: dict, current_index):
        """used for non-first iterations, slice the inputs to length 1."""
        input_ids = model_inputs.pop("input_ids")
        if isinstance(input_ids, Tensor):
            if input_ids.shape[-1] == 1:
                model_inputs["input_ids"] = input_ids
                return
            input_ids = input_ids.asnumpy()

        current_index_tmp = current_index - np.arange(input_ids.size, step=input_ids.shape[1])
        arg = np.arange(input_ids.shape[0])
        inputs_tmp = input_ids[arg, current_index_tmp].reshape(-1, 1)
        model_inputs["input_ids"] = Tensor.from_numpy(inputs_tmp.astype(np.int32))
        return model_inputs

    def _prepare_inputs_for_prefill_flatten(self, input_ids, batch_valid_length, slot_mapping, model_inputs):
        """prepare inputs ids for prefill flatten"""
        batch_valid_length_bs = batch_valid_length.shape[0]  # [bs,]
        input_ids_list = []
        for i in range(batch_valid_length_bs):
            context_len = batch_valid_length[i]
            input_ids_list.append(input_ids[i][:context_len])
        input_ids = np.concatenate(input_ids_list, 0)
        input_ids = input_ids.reshape((1, -1))
        slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)
        return model_inputs

    def _set_block_mgr(self, batch_size, seq_length, num_blocks, block_size):
        """ Set model block table mgr function. """
        if not hasattr(self, "block_mgr") or not self.block_mgr:
            self.block_mgr = BlockTables(num_blocks, block_size, seq_length)

        if self.block_mgr:
            self.block_mgr.init_cache_engine(batch_size)

    def get_valid_length_each_example(self, input_ids, pad_token_id):
        """get valid length and max length in a batch"""
        batch_size = input_ids.shape[0]
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(
                np.max(np.argwhere(input_ids[i] != pad_token_id))
                + 1
            )
        valid_length_each_example = np.array(valid_length_each_example)
        logger.debug("Get the valid for each example is: %s", valid_length_each_example)
        max_length = np.max(valid_length_each_example)
        return valid_length_each_example, max_length
