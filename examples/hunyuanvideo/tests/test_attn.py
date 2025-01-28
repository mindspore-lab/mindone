# flake8: noqa
import math
import os
import sys
import time

import numpy as np
import torch
from easydict import EasyDict as edict
from PIL import Image

import mindspore as ms
from mindspore import amp, ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore

sys.path.insert(0, ".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from hyvideo.modules.attention import FlashAttention, VanillaAttention


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def test_attn_varlen(pt_np=None):
    text_embed_path = "pt_io/text_embed-A-cat-wa.npz"
    data = np.load(text_embed_path)
    prompt_embeds = data["prompt_embeds"]
    prompt_mask = data["prompt_mask"]
    prompt_embeds_2 = data["prompt_embeds_2"]

    attn_in_path = "debug_attn/flash_attn_varlen_inputs.npz"
    attn_in = np.load(attn_in_path)
    q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen = (
        attn_in["q"],
        attn_in["k"],
        attn_in["v"],
        attn_in["cu_seqlens_q"],
        attn_in["cu_seqlens_kv"],
        attn_in["max_seqlen"],
    )

    # prepare ms tensors
    prompt_embeds = ms.Tensor(prompt_embeds)
    text_mask = prompt_mask = ms.Tensor(prompt_mask, dtype=ms.bool_)
    prompt_embeds_2 = ms.Tensor(prompt_embeds_2)

    q = ms.Tensor(q, dtype=ms.bfloat16)
    k = ms.Tensor(k, dtype=ms.bfloat16)
    v = ms.Tensor(v, dtype=ms.bfloat16)

    heads_num = q.shape[-2]
    head_dim = q.shape[-1]
    max_seq_len = q.shape[1]  # img + txt(padded) seqlen

    ## generate mask
    txt_seq_len = text_mask.shape[-1]
    img_seq_len = max_seq_len - txt_seq_len
    # max_seq_len = img_seq_len + txt_seq_len
    bs = q.shape[0]
    mask = ops.ones((bs, 1, 1, max_seq_len), dtype=text_mask.dtype)
    mask[:, 0, 0, img_seq_len:] = text_mask
    mask = mask.tile((1, 1, max_seq_len, 1))  # beginning n columns are all 1
    mask = ops.logical_and(mask, mask.transpose((0, 1, 3, 2)))

    assert int(mask[:, :, 0].sum().asnumpy()) == int(cu_seqlens_q[1])

    # import pdb; pdb.set_trace()
    # cell run
    # compute_attention = FlashAttention(heads_num, head_dim)
    # attn = compute_attention(q, k, v, mask=mask)

    actual_seq_qlen = [int(x) for x in cu_seqlens_q[1:].tolist()]
    actual_seq_kvlen = [int(x) for x in cu_seqlens_kv[1:].tolist()]
    print("D--: ", actual_seq_qlen, actual_seq_kvlen)

    flash_attention = FlashAttentionScore(
        heads_num, keep_prob=1.0, scale_value=1 / math.sqrt(head_dim), input_layout="TND"
    )

    # BSND -> TND
    q = q.reshape((-1, heads_num, head_dim))
    k = k.reshape((-1, heads_num, head_dim))
    v = v.reshape((-1, heads_num, head_dim))
    _, _, _, attn = flash_attention(q, k, v, actual_seq_qlen=actual_seq_qlen, actual_seq_kvlen=actual_seq_kvlen)
    # attn = attn.reshape((bs, max_seq_len, heads_num, head_dim))
    attn = attn.reshape((bs, max_seq_len, heads_num * head_dim))

    # compare output
    attn_out_path = "debug_attn/flash_attn_varlen_output.npy"
    pt_out = np.load(attn_out_path)
    # import pdb; pdb.set_trace()
    print(_diff_res(attn.asnumpy(), pt_out))


if __name__ == "__main__":
    ms.set_context(mode=0)
    test_attn_varlen()
