import sys
import mindspore as ms
from mindspore import amp
import numpy as np
sys.path.append(".")
from janus.models.vq_model import VQ_16
from utils import set_model_param_dtype, diff_res


np.random.seed(42)


def test_decode(pt_ckpt=None, pt_np=None, dtype=ms.float32):
    # shape = (B, C, H, W) = (1, 8, 24, 24)
    # shape = (B, C, H, W) = (1, 8, 12, 12)
    if pt_np:
        pt_data = np.load(pt_np)
        z = pt_data["quant"]
    else:
        z = np.random.normal(size=(B, C, H, W)).astype(np.float32)

    vq = VQ_16()
    if dtype != ms.float32:
        set_model_param_dtype(vq, dtype=dtype, keep_norm_fp32=False)
    if pt_ckpt:
        vq.load_from_checkpoint(pt_ckpt)
    # 
    if dtype != ms.float32:
        amp.auto_mixed_precision(vq, amp_level="O2", dtype=dtype)


    out = vq.decode(ms.Tensor(z))

    print(out.shape)
    print(out.mean(), out.std())

    if pt_np:
        pt_out = pt_data['dec']
        diff = diff_res(out.asnumpy(), pt_out)
        print(diff)

    return out.asnumpy()


def test_encode(pt_ckpt=None, amp=False):
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
    # test_encode()
    test_decode("ckpts/Janus-Pro-1B/pytorch_model.bin", pt_np='tests/vq_dec_io.npz', dtype=ms.bfloat16)

