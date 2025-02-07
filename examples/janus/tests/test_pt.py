import sys
import numpy as np
import time
pt_code_path = "/Users/Samit/Data/Work/HW/ms_kit/aigc/Janus"
sys.path.append(pt_code_path)
import torch
from torch import Tensor
from easydict import EasyDict as edict

from janus.models.vq_model import VQ_16

np.random.seed(42)
torch.manual_seed(42)


def test_decode(pt_ckpt=None, pt_np=None, dtype=torch.float32):
    shape = (B, C, H, W) = (1, 8, 12, 12)
    if pt_np:
        pt_data = np.load(pt_np)
        z = pt_data["quant"]
    else:
        z = np.random.normal(size=(B, C, H, W)).astype(np.float32)

    vq = VQ_16()
    if pt_ckpt is not None:
        sd = torch.load(pt_ckpt)
        pnames = list(sd.keys())
        for p in pnames:
            if not "gen_vision_model" in p:
                sd.pop(p)
            else:
                # remove prefix
                new_pname = p.replace("gen_vision_model.", "")
                sd[new_pname] = sd.pop(p)
        vq.load_state_dict(sd)

    out = vq.decode(Tensor(z))
    np.savez("tests/vq_dec_io.npz",
            quant=z,
            dec=out.detach().cpu().numpy(),
            )

    print(out.shape)
    print(out.sum(), out.std())

    return out.detach().cpu().numpy()


if __name__ == "__main__":
    test_decode("ckpts/Janus-Pro-1B/pytorch_model.bin")
