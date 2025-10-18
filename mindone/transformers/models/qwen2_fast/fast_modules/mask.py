import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from ..utils.utils import is_pynative


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
        chunk_prefill=False
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
            self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                              dtype=compute_type)
        elif use_past and not self.is_pynative:
            if not self.use_flash_attention:
                self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                                  dtype=compute_type)
            elif self.is_dynamic:
                mask_coeff = 1.0 if compute_type is mindspore.bfloat16 else -10000.0
                self.lower_triangle_mask = Tensor(
                    np.triu(np.ones(shape=(128, 128), dtype=np.float16), 1) * mask_coeff, dtype=compute_type
                )
            else:
                self.lower_triangle_mask = None
        else:
            if use_attn_mask_compression:
                if seq_length < 2048:
                    raise ValueError("seq_length should be larger than 2048 when use mask_compression")
                self.lower_triangle_mask = mindspore.Tensor(np.triu(np.ones((2048, 2048), dtype=np.int8), k=1), dtype=mindspore.uint8)
            else:
                self.lower_triangle_mask = Tensor(np.tril(np.ones(shape=(seq_length, seq_length), dtype=np.int8)),
                                                  dtype=compute_type)
        self.shape = P.Shape()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.less_equal = P.LessEqual()
        self.bmm = P.BatchMatMul()
        self.expand_dim = P.ExpandDims()
        self.slice = P.StridedSlice()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.mul_post = P.Mul()
        self.expand_dim_post = P.ExpandDims()
        # seq pp

        self.gather = P.Gather()
        self.seq_split_num = seq_split_num
        self.seq_pipe = seq_split_num > 1
        if self.seq_pipe:
            self.mask_cache = Parameter(Tensor(shape=(batch_size, seq_length), dtype=mindspore.float32, init=Zero()),
                                        name="mask_cache", requires_grad=False, parallel_optimizer=False)
            mask_mask = np.zeros((batch_size, seq_length), dtype=np.int32)
            self.seq_seg_len = seq_length // self.seq_split_num
            for s in range(self.seq_split_num):
                mask_mask[:, s * self.seq_seg_len: (s + 1) * self.seq_seg_len] = s
            self.mask_mask = Tensor(mask_mask)
            self.add_mask = P.Add()
            self.tile_mask = P.Tile()
            self.assign_mask = P.Assign()
            self.mul_mask = P.Mul()
            self.equal_mask = P.Equal()
            np_range = np.arange(seq_length // self.seq_split_num)
            self.seq_seg_range = Tensor(np_range, dtype=mindspore.int32)
            self.seq_seg_len = Tensor(seq_length // self.seq_split_num, dtype=mindspore.int32)
            self.add_seq = P.Add()

    def construct(self, tokens=None, masks=None, seq_chunk=None):
        """Forward process of the CausalMask"""
        if self.use_attn_mask_compression:
            attention_mask = self.lower_triangle_mask
            return attention_mask
        if tokens is not None:
            bs = self.shape(tokens)[0]
            seq_len = self.shape(tokens)[1]
            input_mask = self.cast(self.not_equal(tokens, self.pad_token_id), self.dtype)
        else:
            bs = self.shape(masks)[0]
            seq_len = self.shape(masks)[1]
            input_mask = self.cast(masks, self.dtype)
        shape_right = (bs, 1, seq_len)

        # Mask the padded inputs
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = mask_right

        lower_triangle_mask = self.lower_triangle_mask
        if self.is_pynative or self.is_dynamic:
            lower_triangle_mask = self.slice(self.lower_triangle_mask, (0, 0), (seq_len, seq_len), (1, 1))
        lower_triangle = self.expand_dim(lower_triangle_mask, 0)

        if self.seq_pipe:
            seq_seg_range = self.add_seq(self.seq_seg_range, self.seq_seg_len * seq_chunk)
            attention_mask_chunk = self.gather(lower_triangle, seq_seg_range, 1)
            mask_mask = self.cast(self.equal_mask(self.mask_mask, seq_chunk), self.dtype)
            input_mask = self.tile_mask(input_mask, (1, self.seq_split_num))
            input_mask = self.mul_mask(input_mask, mask_mask)
            input_mask_update = self.add_mask(input_mask, self.mask_cache)
            mask_update = self.assign_mask(self.mask_cache, input_mask_update)
            mask_reshape = self.reshape(input_mask_update, (bs, 1, seq_len * self.seq_split_num))
            mask_reshape = F.depend(mask_reshape, mask_update)
            attention_mask = self.mul(mask_reshape, attention_mask_chunk)
            attention_mask = self.sub(self.one, attention_mask)
            attention_mask = self.expand_dim_post(attention_mask, 1)
            attention_mask = self.cast(attention_mask, mindspore.uint8)
            return attention_mask
        # the returned shape is [bs, 1, seq_length, seq_length]
        attention_mask = self.mul(attention_mask, lower_triangle)
        attention_mask = self.sub(self.one, attention_mask)
        attention_mask = self.expand_dim_post(attention_mask, 1)
        if self.use_flash_attention:
            attention_mask = self.cast(attention_mask, mindspore.uint8)
        else:
            attention_mask = self.mul_post(attention_mask, self.multiply_data)
        return attention_mask

    def prefill(self):
        return self.lower_triangle_mask

    def chunk_masks(self, seq_range):
        masks = self.gather(self.lower_triangle_mask, seq_range, 0)
        return 1 - masks

