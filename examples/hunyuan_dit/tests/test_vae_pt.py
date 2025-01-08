import numpy as np
import torch
from _common import x


def test_vae_pt(ckpt):
    from diffusers.models import AutoencoderKL

    if ckpt is not None:
        net = AutoencoderKL.from_pretrained(ckpt)

    total_params = sum(p.numel() for p in net.parameters())
    print("pt total params: ", total_params)
    print("pt trainable: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    out = net(torch.tensor(x))

    print(out.size())

    return out.numpy()


if __name__ == "__main__":
    out = test_vae_pt("../ckpts/t2i/vae")
    np.save("out_pt_vae.npy", out)
