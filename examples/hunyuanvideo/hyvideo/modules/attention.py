import math
import mindspore as ms
from mindspore import nn, ops


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
    # TODO: add FA here
    # if mode == "flash":
        # if attn_mask is not None and attn_mask.dtype != ms.bool_:
        #    attn_mask = attn_mask.to(q.dtype)
        # x = F.scaled_dot_product_attention(
        #    q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        # )
    # elif mode == "flash":
    #    x = flash_attn_varlen_func(
    #        q,
    #        k,
    #        v,
    #        cu_seqlens_q,
    #        cu_seqlens_kv,
    #        max_seqlen_q,
    #        max_seqlen_kv,
    #    )
    #    # x with shape [(bxs), a, d]
    #    x = x.view(
    #        batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
    #    )  # reshape x to [b, s, a, d]

    # vanilla
    # preapre layout. (B S N D) -> (B N S D)
    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))

    # q: [B N S D)
    scale_factor = 1 / math.sqrt(q.shape[-1])

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

    # attn = (q @ k.transpose(-2, -1)) * scale_factor
    # attn += attn_bias
    # attn = attn.softmax(dim=-1)
    # attn = torch.dropout(attn, p=drop_rate, train=True)
    # x = attn @ v




