import os, sys
import numpy as np
from PIL import Image
import mindspore as ms

sys.path.insert(0, ".")

from hyvideo.modules.models import MMDoubleStreamBlock, MMSingleStreamBlock
from hyvideo.modules.attention import attention


np.random.seed(42)

(B, L, N, D) = (1, 32, 6, 48)
L_txt = 18
hidden_size = N*D

img_shape = (B, L, N*D)
txt_shape = (B, L_txt, N*D)
vec_shape = (B, N*D)
freqs_cis_shape = (L, D)
img_ = np.random.normal(size=img_shape).astype(np.float32)
txt_ = np.random.normal(size=txt_shape).astype(np.float32)
vec_ = np.random.normal(size=vec_shape).astype(np.float32)
freqs_cis_cos_ = np.random.normal(size=freqs_cis_shape).astype(np.float32)
freqs_cis_sin_ = np.random.normal(size=freqs_cis_shape).astype(np.float32)


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def test_attn():
    x_shape = (B, L, N, D) = (1, 32, 6, 48)
    q = np.random.normal(size=x_shape).astype(np.float32)
    k = np.random.normal(size=x_shape).astype(np.float32)
    v = np.random.normal(size=x_shape).astype(np.float32)
    q = ms.Tensor(q)
    k = ms.Tensor(k)
    v = ms.Tensor(v)

    out = attention(q, k, v, mode='vanilla')

    print(out.shape)


def test_dualstream_block():
    img = ms.Tensor(img_)
    txt = ms.Tensor(txt_)
    vec = ms.Tensor(vec_)
    freqs_cis = (ms.Tensor(freqs_cis_cos_), ms.Tensor(freqs_cis_sin_))
    # freqs_cis_cos = ms.Tensor(freqs_cis_cos)
    # freqs_cis_sin = ms.Tensor(freqs_cis_sin)

    print('input sum: ', img.sum())
    block = MMDoubleStreamBlock(
        hidden_size = hidden_size,
        heads_num=N,
        mlp_width_ratio=1,
        qkv_bias=True,
    )

    img_out, txt_out = block(img, txt, vec, freqs_cis=freqs_cis)
    # out = block(img, txt, vec, freqs_cis_cos=freqs_cis_cos, freqs_cis_sin=freqs_cis_sin)
    print(img_out.shape)
    print(img_out.mean(), img_out.std())  # -0.00013358534 1.0066756 for fp32
    print(txt.shape)
    print(txt_out.mean(), txt_out.std())


def test_singlestream_block(pt_fp: str=None):
    img = ms.Tensor(img_)
    txt = ms.Tensor(txt_)
    x = ms.ops.concat([img, txt], 1)
    vec = ms.Tensor(vec_)
    freqs_cis = (ms.Tensor(freqs_cis_cos_), ms.Tensor(freqs_cis_sin_))

    print('input sum: ', x.sum())
    block = MMSingleStreamBlock(
        hidden_size = hidden_size,
        heads_num=N,
        mlp_width_ratio=1,
    )

    out = block(x, vec, L_txt, freqs_cis=freqs_cis)

    print(out.shape)
    print(out.mean(), out.std())

    if pt_fp:
        pt_out = np.load(pt_fp)
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)


if __name__ == "__main__":
    ms.set_context(mode=0, jit_syntax_level=ms.STRICT)
    # test_attn()
    # test_dualstream_block()
    test_singlestream_block('tests/pt_single_stream.npy')
