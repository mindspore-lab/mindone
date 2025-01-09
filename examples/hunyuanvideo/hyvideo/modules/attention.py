import math
import mindspore as ms
from mindspore import nn, ops


class VanillaAttention(nn.Cell):
    def __init__(self, head_dim):
        super().__init__()
        self.scale_factor = 1 / math.sqrt(head_dim)

    def construct(q, k, v):
       # preapre layout. (B S N D) -> (B N S D)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # q: [B N S D)
        b, a, s, _ = q.shape
        attn = ops.bmm(q, k.transpose(0, 1, 3, 2)) * self.scale_factor
        # attn= attn.to(ms.float32)  # (B N Sq Sk)
        attn = ops.softmax(attn, axis=-1)
        x = ops.bmm(attn.to(v.dtype), v)  # (B N S D)

        # prepare output layout
        x = ops.transpose(x, (0, 2, 1, 3))

        b, s, a, d = x.shape
        out = x.reshape(b, s, -1)

        return out


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
):
    '''
    q/k/v: (B S N D)
    attn_mask: (B Sq Sk)

    output: (B S H)
    '''
    # vanilla
    # preapre layout. (B S N D) -> (B N S D)
    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))

    # q: [B N S D)

    b, a, s, _ = q.shape

    # assert attn_mask==None, 'not supported'

    # TODO: Maybe force q and k to be float32 to avoid numerical overflow
    attn = ops.bmm(q, k.transpose(0, 1, 3, 2)) * scale_factor

    # attn= attn.to(ms.float32)  # (B N Sq Sk)
    attn = ops.softmax(attn, axis=-1)
    x = ops.bmm(attn.to(v.dtype), v)  # (B N S D)

    # prepare output layout
    x = ops.transpose(x, (0, 2, 1, 3))

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)

    return out
