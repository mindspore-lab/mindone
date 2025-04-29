"""Infer Attention Layer"""
import math

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell

from .paged_attention_fa_block import FlashAttention
from .paged_attention_mgr import PagedAttentionMgr


class InferRotaryEmbedding(Cell):
    r"""
    Infer Rotary Position Embedding.

    Args:
            rotary_cos_format (int): - Choose the rotary embedding cos format. Default 0.

    Inputs:
            query: the query matrix.
            key: the key matrix.
            freqs_cis: The precompute freqs and mask for rotary position embedding used in attention.
            batch_valid_length: Int32 tensor with shape [batch_size] the past calculated the index.

    Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, rotary_cos_format=0):
        super().__init__()
        self.rotary_cos_format = rotary_cos_format
        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(self.rotary_cos_format)

    def construct(self, query: Tensor, key: Tensor, freqs_cis, batch_valid_length):
        """Forward of rotary position embedding."""
        freqs_cos, freqs_sin, _ = freqs_cis
        return self.rotary_embedding_op(query, key, freqs_cos, freqs_sin, batch_valid_length)


class InferAttention(Cell):
    """Infer Attention Layer.

    This function contains the InferAttention primitives used with FlashAttention and PagedAttention for kbk infer.

    B -- Batch size
    S1 -- Sequence length of query. The value ranges from 1 to 32768 and is a multiple of 16.
    S2 -- Sequence length of key and value. The value ranges from 1 to 32768 and is a multiple of 16.
    N1 -- Num heads of query
    N2 -- Num heads of key and value, and N2 must be a factor of N1
    D -- Head size. Support value: 64, 80, 96, 120, 128 and 256.
    H1 -- Hidden size of query, which equals to N1 * D
    H2 -- Hidden size of key and value, which equals to N2 * D

    Args:
        n_head (int): The head num of query.
        head_dim (int): The dim of head.
        n_kv_head (int): The head num of key and value.
        pa_n_head_split (int): The query head num of paged attention op after split.
        pa_n_kv_head_split (int): The key and value head num of paged attention op after split.
        keep_prob (float): The keep probability of dropout. Default: 1.0.
        scale_value (float): The scale factor of score. Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        Default: "BSH".
        sparse_mode (int): Indicates sparse mode. Default 0.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
              matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
              be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required..
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
              each Batch axis is different. Not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.
        block_size (int): Block size for paged attention.
        num_blocks (int): Block num for paged attention.
        use_flash_attention (bool): The value is True if chosen to use flash attention. Default: True.
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        use_rope_rotary_emb (bool): If use rotary embedding. Default True.
        rotary_cos_format (int): Choose the rotary embedding cos format. Default 0.
        rotary_dtype (mstype): Compute dtype for rope op. Default mstype.float16.
        compute_dtype (mstype): Compute dtype for infer attention. Default mstype.float16.
        parallel_decoding (mstype): If open parallel decoding. Default False.

    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
          Input tensor of shape :math:`(B, S1, H1)` or :math:`(B, N1, S1, D)`.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
          Input tensor of shape :math:`(B, S2, H2)` or :math:`(B, N2, S2, D)`.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
          Input tensor of shape :math:`(B, S2, H2)` or :math:`(B, N2, S2, D)`.
        - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
          Used for incremental prediction when the use_past is True. Default None.
        - **block_tables** (Tensor[int64]) - Store mapping tables for each sequence.
        - **slot_mapping** (Tensor[int32]) - Store token cache physical slot index.
        - **freqs_cos** (Tensor[float16, bfloat16]) - The precompute freqs cos for rotary position embedding used in
          attention, shape is (seq_len, head_dim).
        - **freqs_sin** (Tensor[float16, bfloat16]) - The precompute freqs sin for rotary position embedding used in
          attention, shape is (seq_len, head_dim).
        - **attn_mask** (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, :math:`(B, 1, S1, S2)`,
          :math:`(S1, S2)` or (2048, 2048).
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization.
          Input tensor of shape :math:`(B, N1, S1, S2)`, :math:`(1, N1, S1, S2)`, :math:`(B, N1, 1024, S2)`,
          :math:`(1, N1, 1024, S2)` or (1024, 1024).
        - **prefix_keys_values** (Union[Tensor[float16, bfloat16], None]) - The prefix keys values.
        - **q_seq_lens** (Union[Tensor[int32], None]) - The query actual seq len.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend910B``
    """

    def __init__(
        self,
        n_head,
        head_dim,
        n_kv_head,
        pa_n_head_split=None,
        pa_n_kv_head_split=None,
        keep_prob=1.0,
        scale_value=1.0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        sparse_mode=0,
        block_size=16,
        num_blocks=1024,
        seq_length=-1,
        is_dynamic=True,
        use_flash_attention=True,
        use_alibi_mask=False,
        use_rope_rotary_emb=True,
        rotary_cos_format=0,
        compute_dtype=mstype.float16,
        parallel_decoding=False,
        chunk_prefill=False,
    ):
        super(InferAttention, self).__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_kv_head = n_kv_head
        self.pa_n_head_split = pa_n_head_split if pa_n_head_split is not None else n_head
        self.pa_n_kv_head_split = pa_n_kv_head_split if pa_n_kv_head_split is not None else n_kv_head
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.sparse_mode = sparse_mode
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_flash_attention = use_flash_attention
        self.use_alibi_mask = use_alibi_mask
        self.use_rope_rotary_emb = use_rope_rotary_emb
        self.rotary_cos_format = rotary_cos_format
        self.compute_dtype = compute_dtype
        self.is_first_iteration = True
        self.chunk_prefill = chunk_prefill
        self.inv_norm_factor = Tensor(1.0 / math.sqrt(self.head_dim), dtype=compute_dtype)
        self.n_rep = self.n_head // self.n_kv_head
        self.use_attention_mask = True
        self.is_dynamic = is_dynamic

        if self.is_dynamic:
            self.input_layout = "TH"
            self.use_attention_mask = not self.use_alibi_mask
        else:
            self.input_layout = "BSH"
            self.use_attention_mask = False

        self.flash_attention = FlashAttention(
            head_num=self.n_head,
            pre_tokens=self.pre_tokens,
            next_tokens=self.next_tokens,
            keep_prob=self.keep_prob,
            scale_value=self.scale_value,
            sparse_mode=self.sparse_mode,
            use_attention_mask=self.use_attention_mask,
            use_alibi_mask=self.use_alibi_mask,
            input_layout=self.input_layout,
        )

        kv_shape = (self.num_blocks, self.block_size, self.n_kv_head, self.head_dim)
        self.paged_attention_mgr = PagedAttentionMgr(
            self.pa_n_head_split,
            self.head_dim,
            self.pa_n_kv_head_split,
            kv_shape,
            seq_length,
            compute_dtype=self.compute_dtype,
            parallel_decoding=parallel_decoding,
            chunk_prefill=chunk_prefill,
        )
        if use_rope_rotary_emb:
            self.rotary_embedding = InferRotaryEmbedding(self.rotary_cos_format)

    def _apply_rotary_pos_emb(self, query, key, freqs_cis, batch_valid_length):
        """
        apply rotary pos embedding

        Inputs:
            query: the query matrix
            key: the key matrix
            freqs_cis: The precompute freqs and mask for rotary position embedding used in attention.
            batch_valid_length: Int32 tensor with shape [batch_size] the past calculated the index.
        Outputs:
            query: the query matrix
            key: the key matrix
        """
        return self.rotary_embedding(query, key, freqs_cis, batch_valid_length)  # dp, mp, 1, 1

    def _prefill_attention(self, query, key, value, attn_mask, alibi_mask, actual_seq_qlen=None, actual_seq_kvlen=None):
        """
        prefill attention
        """
        if self.input_layout == "TH":
            if self.use_flash_attention:
                # [1, actual_seq_len, H]
                bs, seq_len, _ = query.shape
                # [1, actual_seq_len, H] -> [actual_seq_len, H]
                query = query.reshape((-1, self.n_head * self.head_dim))
                key = key.reshape((-1, self.n_kv_head * self.head_dim))
                value = value.reshape((-1, self.n_kv_head * self.head_dim))
                # [actual_seq_len, H]
                output = self.flash_attention(
                    query, key, value, attn_mask, alibi_mask, None, None, actual_seq_qlen, actual_seq_kvlen
                )
                # [actual_seq_len, H] -> [1, actual_seq_len, H]
                output = output.reshape((bs, seq_len, self.n_head * self.head_dim))
                return output

        if self.input_layout == "BSH":
            if self.use_flash_attention:
                # query shape:(B, S, H), key shape:(B, S, H), value shape:(B, S, H)
                return self.flash_attention(query, key, value, attn_mask, alibi_mask)

        raise ValueError("FlashAttention input layout:{} is not supported.".format(self.input_layout))

    def _incre_attention(
        self, query, batch_valid_length, block_tables, alibi_mask=None, attn_mask=None, q_seq_lens=None
    ):
        if self.use_alibi_mask:
            return self.paged_attention_mgr.paged_attn_with_alibi(query, batch_valid_length, block_tables, alibi_mask)
        return self.paged_attention_mgr.paged_attn(
            query, batch_valid_length, block_tables, attn_mask=attn_mask, q_seq_lens=q_seq_lens
        )

    def construct(
        self,
        query,
        key,
        value,
        batch_valid_length,
        block_tables,
        slot_mapping,
        freqs_cis=None,
        attn_mask=None,
        alibi_mask=None,
        prefix_keys_values=None,
        q_seq_lens=None,
    ):
        """Forward process of the Infer Attention Cell"""
        if self.use_rope_rotary_emb:
            query, key = self._apply_rotary_pos_emb(query, key, freqs_cis, batch_valid_length)

        self.paged_attention_mgr(key, value, slot_mapping, batch_valid_length)

        if self.is_first_iteration:
            return self._prefill_attention(
                query, key, value, attn_mask, alibi_mask, batch_valid_length, batch_valid_length
            )
        else:
            return self._incre_attention(query, batch_valid_length, block_tables, alibi_mask, attn_mask, q_seq_lens)
