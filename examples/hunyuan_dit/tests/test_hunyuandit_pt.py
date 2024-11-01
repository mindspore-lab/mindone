import sys

import numpy as np
import torch
from _common import inputs, kwargs


def test_hunyuan_dit_pt(ckpt):
    pt_code_path = "/HunyuanDiT_pt/HunyuanDiT/"
    sys.path.insert(0, pt_code_path)
    from hydit.config import get_args
    from hydit.modules.models import HUNYUAN_DIT_CONFIG, HunYuanDiT

    model_config = HUNYUAN_DIT_CONFIG["DiT-XL/2"]
    args = get_args()
    cuda = torch.device("cuda:0")
    net = HunYuanDiT(args, **model_config).half().to(cuda)
    net.eval()

    if ckpt is not None:
        sd = torch.load(ckpt)
        net.load_state_dict(sd)

    total_params = sum(p.numel() for p in net.parameters())
    print("pt total params: ", total_params)
    print("pt trainable: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    inputs = [torch.tensor(input).to(cuda) for input in inputs]
    kwargs = {k: torch.tensor(v).to(cuda) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    out = net(*inputs, **kwargs)["x"]

    print(out.size())

    return out.cpu().detach.numpy()


if __name__ == "__main__":
    out = test_hunyuan_dit_pt("../ckpts/t2i/model/pytorch_model_distill.pt")
    np.save("out_pt_hunyuan_dit.npy", out)
