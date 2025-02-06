import os
import sys
import mindspore as ms
import numpy as np
sys.path.append(".")
from janus.models.vq_model import VQ_16

def test_decode(ckpt_path=None, amp=False):
    # shape = (B, C, H, W) = (1, 8, 24, 24)
    shape = (B, C, H, W) = (1, 8, 12, 12)
    vq = VQ_16()
    z = np.random.normal(size=(B, C, H, W)).astype(np.float32)
    out = vq.decode(ms.Tensor(z))

    print(out.shape)
    print(out.mean(), out.std())

    return out.asnumpy()


def test_encode(ckpt_path=None, amp=False):
    # shape = (B, C, H, W) = (1, 8, 24, 24)
    shape = (B, C, H, W) = (1, 3, 64, 64)
    vq = VQ_16()
    x = np.random.normal(size=(B, C, H, W)).astype(np.float32)
    out = vq.encode(ms.Tensor(x))[0]

    print(out.shape)
    print(out.mean(), out.std())

    return out.asnumpy()



if __name__ == '__main__':
    ms.set_context(mode=1)
    test_encode()
    # test_decode()

