# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Paged Attention Manager for inference."""
import math

import mindspore.common.dtype as mstype
from mindspore import nn, Parameter
from mindspore import ops as P
from mindspore.common.initializer import initializer


class PagedAttentionMgr(nn.Cell):
    """Paged Attention Manager."""

    def __init__(self,
                 n_heads,
                 head_dim,
                 n_kv_heads,
                 kv_shape,
                 seq_length=-1,
                 compute_dtype=mstype.float16,
                 parallel_decoding=False,
                 chunk_prefill=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.seq_length = seq_length
        self.is_first_iteration = True
        self.scale_value = 1 / math.sqrt(self.head_dim)
        self.key_cache = Parameter(initializer('zeros', kv_shape, compute_dtype), name="key_cache",
                                   requires_grad=False)
        self.value_cache = Parameter(initializer('zeros', kv_shape, compute_dtype), name="value_cache",
                                     requires_grad=False)

        self.reshape_and_cache = P.auto_generate.ReshapeAndCache()
        self.paged_attention = P.auto_generate.PagedAttention(self.n_heads,
                                                              self.scale_value,
                                                              self.n_kv_heads)
        self.paged_attention_with_alibi = P.auto_generate.PagedAttentionMask(self.n_heads,
                                                                             self.scale_value,
                                                                             self.n_kv_heads)
        self.parallel_decoding = parallel_decoding
        self.chunk_prefill = chunk_prefill

    # pylint: disable=W0613
    def construct(self, key, value, slot_mapping, batch_valid_length=None):
        """The forward compute of KVCache for Paged Attention."""
        return self.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping)

    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        if self.parallel_decoding or self.chunk_prefill:
            attn_mask = attn_mask.astype(mstype.bool_).astype(query.dtype) * -10000
            return self.paged_attention(query, self.key_cache, self.value_cache, block_tables, batch_valid_length,
                                        None, None, attn_mask, q_seq_lens)
        return self.paged_attention(query, self.key_cache, self.value_cache, block_tables, batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self.paged_attention_with_alibi(query, self.key_cache, self.value_cache,
                                               block_tables, batch_valid_length, None, None, alibi_tensor)

