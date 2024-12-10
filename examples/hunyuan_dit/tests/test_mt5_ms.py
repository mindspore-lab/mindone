import sys

import numpy as np
from _common import x

import mindspore as ms


def test_mt5_ms(ckpt):
    ms_code_path = "PATH/mindone/examples/hunyuan_dit/"
    sys.path.insert(0, ms_code_path)
    from hydit.modules.text_encoder import MT5Embedder

    if ckpt is not None:
        net = MT5Embedder(ckpt, mindspore_dtype=ms.float16, max_length=256)

    total_params = sum([param.size for param in net.get_parameters()])
    print("ms total params: ", total_params)
    print("ms trainable: ", sum([param.size for param in net.get_parameters() if param.requires_grad]))

    out = net(ms.tensor(x.input_ids), ms.tensor(x.attention_mask))

    print(out.shape)

    return out.asnumpy()


def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / np.fabs(pt_val).mean()
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


if __name__ == "__main__":
    ms.set_context(mode=0)
    ms_out = test_mt5_ms("../ckpts/t2i/mt5")
    np.save("out_ms_mt5.npy", ms_out)

    pt_out = np.load("out_pt_mt5.npy")
    print(_diff_res(ms_out, pt_out))
