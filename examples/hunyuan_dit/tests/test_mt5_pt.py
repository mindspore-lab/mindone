import sys

import numpy as np
import torch
from _common import x


def test_mt5_pt(ckpt):
    pt_code_path = "/HunyuanDiT_pt/HunyuanDiT/"
    sys.path.insert(0, pt_code_path)
    from hydit.modules.text_encoder import MT5Embedder

    if ckpt is not None:
        net = MT5Embedder(ckpt, torch_dtype=torch.float16, max_length=256)

    total_params = sum(p.numel() for p in net.parameters())
    print("pt total params: ", total_params)
    print("pt trainable: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    out = net(torch.tensor(x))

    print(out.size())

    return out.numpy()


if __name__ == "__main__":
    out = test_mt5_pt("../ckpts/t2i/mt5")
    np.save("out_pt_mt5.npy", out)
