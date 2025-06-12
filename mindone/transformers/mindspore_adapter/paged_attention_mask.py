import numpy as np

import mindspore
from mindspore import Tensor, context, nn
from mindspore.ops import operations as P


def is_pynative():
    mode = context.get_context("mode")
    return mode == context.PYNATIVE_MODE


class LowerTriangularMaskWithDynamic(nn.Cell):
    r"""
    Get the Strictly Lower triangular matrix from the input_ids.
    """

    def __init__(
        self,
        seq_length,
        batch_size=1,
        compute_type=mindspore.float16,
        is_dynamic=False,
        pad_token_id=0,
        use_flash_attention=False,
        use_attn_mask_compression=False,
        use_past=False,
        seq_split_num=1,
        chunk_prefill=False,
    ):
        super().__init__()
        self.dtype = compute_type
        self.is_dynamic = is_dynamic
        self.pad_token_id = pad_token_id
        self.use_flash_attention = use_flash_attention
        self.use_attn_mask_compression = use_attn_mask_compression
        self.seq_length = seq_length
        self.is_first_iteration = True
        self.multiply_data = Tensor([-10000.0], dtype=compute_type)
        self.one = Tensor([1.0], dtype=compute_type)
        self.is_pynative = is_pynative()
        self.chunk_prefill = chunk_prefill
        if use_past and chunk_prefill:
            self.lower_triangle_mask = Tensor(
                np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)), dtype=compute_type
            )
        elif use_past and not self.is_pynative:
            if not self.use_flash_attention:
                self.lower_triangle_mask = Tensor(
                    np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)), dtype=compute_type
                )
            elif self.is_dynamic:
                mask_coeff = 1.0 if compute_type == mindspore.bfloat16 else -10000.0
                self.lower_triangle_mask = Tensor(
                    np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * mask_coeff, dtype=compute_type
                )
            else:
                self.lower_triangle_mask = None
        else:
            if use_attn_mask_compression:
                if seq_length < 2048:
                    raise ValueError("seq_length should be larger than 2048 when use mask_compression")
                self.lower_triangle_mask = mindspore.Tensor(
                    np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=mindspore.uint8
                )
            else:
                self.lower_triangle_mask = Tensor(
                    np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)), dtype=compute_type
                )
        self.not_equal = P.NotEqual()
        self.less_equal = P.LessEqual()
        self.bmm = P.BatchMatMul()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.sub = P.Sub()
        self.expand_dim_post = P.ExpandDims()
        # seq pp

        self.gather = P.Gather()
        self.seq_split_num = seq_split_num
        self.seq_pipe = seq_split_num > 1

    def construct(self, tokens=None, masks=None, seq_chunk=None):
        """Forward process of the CausalMask"""
        if self.use_attn_mask_compression:
            attention_mask = self.lower_triangle_mask
            return attention_mask
        if tokens is not None:
            bs = tokens.shape[0]
            seq_len = tokens.shape[1]
            input_mask = self.not_equal(tokens, self.pad_token_id).to(self.dtype)
        else:
            bs = masks.shape[0]
            seq_len = masks.shape[1]
            input_mask = masks.to(self.dtype)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = input_mask.reshape(shape_right)
        attention_mask = mask_right

        lower_triangle_mask = self.lower_triangle_mask
        if self.is_pynative or self.is_dynamic:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
        lower_triangle = self.expand_dim(lower_triangle_mask, 0)

        # the returned shape is [bs, 1, seq_length, seq_length]
        attention_mask = attention_mask * lower_triangle
        attention_mask = self.sub(self.one, attention_mask)
        attention_mask = self.expand_dim_post(attention_mask, 1)
        if self.use_flash_attention:
            attention_mask = attention_mask.to(mindspore.uint8)
        else:
            attention_mask = attention_mask * self.multiply_data
        return attention_mask

    def prefill(self):
        return self.lower_triangle_mask

    def chunk_masks(self, seq_range):
        masks = self.gather(self.lower_triangle_mask, seq_range, 0)
        return 1 - masks
