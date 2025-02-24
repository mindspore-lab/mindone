import mindspore as ms
from mindspore import nn

from ..mindspore_adapter.hybrid_attention import LongContextAttention
from ..mindspore_adapter.scaled_dot_product_attn import scaled_dot_product_attention


class Attention(nn.Cell):
    def __init__(self, sp_group: str = None, head_dim: int = None, head_num: int = None):
        super().__init__()
        self.hybrid_seq_parallel_attn = LongContextAttention(
            ring_pg=None, ulysses_pg=sp_group, head_dim=head_dim, head_num=head_num
        )

    def attn_processor(self, attn_type):
        if attn_type == "mindspore":
            return self.mindspore_attn_func
        elif attn_type == "parallel":
            return self.parallel_attn_func
        else:
            raise Exception("Not supported attention type...")

    def mindspore_attn_func(self, q, k, v, attn_mask=None, drop_rate=0.0, **kwargs):
        if attn_mask is not None and attn_mask.dtype != ms.bool_:
            attn_mask = attn_mask.to(q.dtype)

        if attn_mask is not None and attn_mask.ndim == 3:  # no head
            n_heads = q.shape[2]
            attn_mask = attn_mask.unsqueeze(1).tile((1, n_heads, 1, 1))

        # q, k, v = map(lambda x: rearrange(x, 'b s h d -> b h s d'), (q, k, v))
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        x = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate)

        # x = rearrange(x, 'b h s d -> b s h d')
        x = x.transpose(0, 2, 1, 3)

        return x

    def parallel_attn_func(
        self,
        q,
        k,
        v,
        **kwargs
        # no mask, no causal
    ):
        x = self.hybrid_seq_parallel_attn(q, k, v)
        return x
