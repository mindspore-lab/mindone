import sys
import numpy as np
from PIL import Image
import mindspore as ms

sys.path.insert(0, ".")

from hyvideo.modules.models import MMDoubleStreamBlock

np.random.seed(42)


def test_dualstream_block():
    (B, L, N, D) = (1, 32, 6, 48)
    L_txt = 18
    hidden_size = N*D

    img_shape = (B, L, N*D)
    txt_shape = (B, L_txt, N*D)
    vec_shape = (B, N*D)
    freqs_cis_shape = (L, D)
    img = np.random.normal(size=img_shape).astype(np.float32)
    txt = np.random.normal(size=txt_shape).astype(np.float32)
    vec = np.random.normal(size=vec_shape).astype(np.float32)
    freqs_cis_cos = np.random.normal(size=freqs_cis_shape).astype(np.float32)
    freqs_cis_sin = np.random.normal(size=freqs_cis_shape).astype(np.float32)

    img = ms.Tensor(img)
    txt = ms.Tensor(txt)
    vec = ms.Tensor(vec)
    freqs_cis = (ms.Tensor(freqs_cis_cos), ms.Tensor(freqs_cis_sin))

    block = MMDoubleStreamBlock(
        hidden_size = hidden_size,
        heads_num=N,
        mlp_width_ratio=1,
        qkv_bias=True,
    )


    img_out, txt_out = block(img, txt, vec, freqs_cis=freqs_cis)
    # out = block(img, txt, vec, freqs_cis=freqs_cis)
    # print(out.shape)

    print(img_out.shape)
    print(img_out.mean(), img_out.std())
    print(txt.shape)
    print(txt_out.mean(), txt_out.std())


if __name__ == "__main__":
    ms.set_context(mode=1)

    test_dualstream_block()
