import numpy as np
from _common import x

import mindspore as ms


def test_vae_ms(ckpt):
    from mindone.diffusers.models import AutoencoderKL

    if ckpt is not None:
        net = AutoencoderKL.from_pretrained(ckpt)

    total_params = sum([param.size for param in net.get_parameters()])
    print("ms total params: ", total_params)
    print("ms trainable: ", sum([param.size for param in net.get_parameters() if param.requires_grad]))

    out = net(ms.tensor(x))

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
    ms_out = test_vae_ms("../ckpts/t2i/vae")
    np.save("out_ms_vae.npy", ms_out)

    pt_out = np.load("out_pt_vae.npy")
    print(_diff_res(ms_out, pt_out))
